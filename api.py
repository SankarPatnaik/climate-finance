from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
DATA_PATH = Path(os.getenv("CLIMATE_DATA_PATH", BASE_DIR / "data" / "climate_finance_sample.csv"))

app = FastAPI(
    title="Climate Finance Data API",
    version="1.0.0",
    description="Queryable API exposing climate finance indicators with optional chat support.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class YearValue(BaseModel):
    year: str
    value: float


class ClimateIndicator(BaseModel):
    Country: str
    Indicator: str
    Unit: str
    years: List[YearValue]


class ClimateGroupedIndicator(BaseModel):
    Country: str
    Indicator: str
    Sector: str
    Unit: str
    years: List[YearValue]


class ClimateSector(BaseModel):
    Country: str
    Sector: str
    Unit: str
    years: List[YearValue]


class ClimateGroupedSector(BaseModel):
    Country: str
    Sector: str
    Indicator: str
    Unit: str
    years: List[YearValue]


class ClimateCountry(BaseModel):
    Country: str
    ISO3: str
    years: List[YearValue]


def _load_dataset() -> tuple[pd.DataFrame, List[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)
    year_columns = [col for col in df.columns if col.isdigit()]

    required = {"Country", "ISO3", "Sector", "Indicator", "Unit"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {', '.join(sorted(missing))}")

    return df.fillna(0), year_columns


def _format_record(record: dict, year_columns: List[str]) -> dict:
    years = [
        {
            "year": column,
            "value": float(record.get(column, 0)),
        }
        for column in year_columns
    ]

    base_keys = {k: str(v) for k, v in record.items() if k not in year_columns}
    base_keys["years"] = years
    return base_keys


def _group_and_transform(
    df: pd.DataFrame,
    year_columns: List[str],
    group_fields: List[str],
    country: Optional[str] = None,
    sector: Optional[str] = None,
    indicator: Optional[str] = None,
) -> List[dict]:
    filtered = df.copy()

    if country:
        filtered = filtered[filtered["Country"] == country]
    if sector:
        filtered = filtered[filtered["Sector"] == sector]
    if indicator:
        filtered = filtered[filtered["Indicator"] == indicator]

    grouped = filtered.groupby(group_fields + ["Unit"], dropna=False)[year_columns].sum().reset_index()
    return [_format_record(record, year_columns) for record in grouped.to_dict(orient="records")]


def _get_chat_model() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured in the environment")
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)


def _get_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Provide concise insights about climate and environment information using the user's context.",
            ),
            ("user", "{query}"),
        ]
    )


df, YEAR_COLUMNS = _load_dataset()


@app.get("/", response_class=HTMLResponse)
def landing_page() -> HTMLResponse:
    index_file = BASE_DIR / "static" / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=500, detail="Landing page not found")
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/sector", response_model=List[ClimateSector], tags=["Sector"])
async def get_sector(
    country: Optional[str] = Query(None, description="Filter by Country"),
    sector: Optional[str] = Query(None, description="Filter by Sector"),
) -> List[dict]:
    """Retrieve aggregated sector data for each country."""

    return _group_and_transform(df, YEAR_COLUMNS, ["Country", "Sector"], country=country, sector=sector)


@app.get("/group/sector", response_model=List[ClimateGroupedSector], tags=["Sector"])
async def get_grouped_sector(
    country: Optional[str] = Query(None, description="Filter by Country"),
    sector: Optional[str] = Query(None, description="Filter by Sector"),
) -> List[dict]:
    """Retrieve aggregated sector data grouped by indicator."""

    return _group_and_transform(
        df, YEAR_COLUMNS, ["Country", "Indicator", "Sector"], country=country, sector=sector
    )


@app.get("/indicator", response_model=List[ClimateIndicator], tags=["Indicator"])
async def get_indicator(
    country: Optional[str] = Query(None, description="Filter by Country"),
    indicator: Optional[str] = Query(None, description="Filter by Indicator"),
) -> List[dict]:
    """Retrieve aggregated indicator data for each country."""

    return _group_and_transform(df, YEAR_COLUMNS, ["Country", "Indicator"], country=country, indicator=indicator)


@app.get("/group/indicator", response_model=List[ClimateGroupedIndicator], tags=["Indicator"])
async def get_grouped_indicator(
    country: Optional[str] = Query(None, description="Filter by Country"),
    indicator: Optional[str] = Query(None, description="Filter by Indicator"),
) -> List[dict]:
    """Retrieve aggregated indicator data grouped by sector."""

    return _group_and_transform(
        df, YEAR_COLUMNS, ["Country", "Sector", "Indicator"], country=country, indicator=indicator
    )


@app.get("/country", response_model=List[ClimateCountry], tags=["Country"])
async def get_country(country: Optional[str] = Query(None, description="Filter by Country")) -> List[dict]:
    """Retrieve aggregated values grouped by country and ISO3 code."""

    return _group_and_transform(df, YEAR_COLUMNS, ["Country", "ISO3"], country=country)


@app.get("/chat", tags=["Chatbot"])
async def chat(query: str = Query(..., description="User query")) -> dict:
    """Lightweight chat endpoint backed by Groq's LLM."""

    model = _get_chat_model()
    prompt = _get_prompt().invoke({"query": query})
    response = model.invoke(prompt)
    return {"response": response.content}


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", port=5000, log_level="info", reload=True)
