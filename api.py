from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from typing import List
import pandas as pd
import uvicorn
import getpass
import os
os.environ["GROQ_API_KEY"] = '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'


prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant, your task is to provide information regrading climate and environment information"),
    ("user", "{query}")
])


from langchain_groq import ChatGroq

model = ChatGroq(model="llama-3.3-70b-versatile")


class Year(BaseModel):
    year : str
    value : float



class ClimateIndicator(BaseModel):
    Country: str
    Indicator: str
    Unit: str
    years: List[Year]

class ClimategroupIndicator(BaseModel):
    Country: str
    Indicator: str
    Sector:str
    Unit: str
    years: List[Year]


class ClimateSector(BaseModel):
    Country: str
    Sector: str
    Unit: str
    years: List[Year]  

class ClimategroupSector(BaseModel):
    Country: str
    Sector: str
    Indicator : str
    Unit: str
    years: List[Year]  


class ClimateCountry(BaseModel):
    Country: str
    ISO3: str
    years: List[Year]  

df = pd.read_csv('test.csv')




# --- FastAPI Endpoints ---

app = FastAPI(
    title="Climate Finance Data API",
    version="1.0.0",
    description="API for querying Climate Finance data.",
)



app = FastAPI(
    title="Climate Finance Data API",
    version="1.0.0",
    description="API for querying Climate Finance data.",
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def transform(data):
    year_list = []
    new_data ={}
    for key , val in data.items():
        if key.isdigit():
            year_list.append({'year':key,'value':val})
        else:
            new_data[key]=str(val)
    new_data['years']=year_list
    return new_data

#response_model=List[ClimateSector]
#response_model=List[ClimateIndicator]
@app.get("/sector", response_model=List[ClimateSector],tags=["Sector"])
async def getSectorResponse(
    country:str = Query(None, description="Filter by Country"),
    sector : str = Query(None, description="Filter by Sector")):

    """Retrieve all sector response records"""
    df_temp = df.groupby(['Country','Sector','Unit'])[['2005',
       '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
       '2015', '2016', '2017', '2018', '2022']].sum().reset_index()


    if country:
        df_temp = df_temp[df_temp['Country']==country]

    if sector:
        df_temp = df_temp[df_temp['Sector']==sector]

    df_temp = df_temp.fillna(0)
    new_data = [transform(record) for record in df_temp[:1000].to_dict(orient='records')]
    return new_data

@app.get("/group/sector", response_model=List[ClimategroupSector],tags=["Sector"])
async def getSectorResponse(
    country:str = Query(None, description="Filter by Country"),
    sector : str = Query(None, description="Filter by Sector")):

    """Retrieve all sector response records"""
    df_temp = df.groupby(['Country','Indicator','Sector','Unit'])[['2005',
       '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
       '2015', '2016', '2017', '2018', '2022']].sum().reset_index()


    if country:
        df_temp = df_temp[df_temp['Country']==country]

    if sector:
        df_temp = df_temp[df_temp['Sector']==sector]

    df_temp = df_temp.fillna(0)
    new_data = [transform(record) for record in df_temp[:1000].to_dict(orient='records')]
    return new_data

#response_model=List[ClimateIndicator] 
@app.get("/indicator",response_model=List[ClimateIndicator] ,tags=["Indicator"])
async def getIndicatorResponse(
    country:str = Query(None, description="Filter by Country"),
    indicator : str = Query(None, description="Filter by Indicator")):

    """Retrieve all Indicator response records"""
    df_temp = df.groupby(['Country','Indicator','Unit'])[['2005',
       '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
       '2015', '2016', '2017', '2018', '2022']].sum().reset_index()


    if country:
        df_temp = df_temp[df_temp['Country']==country]

    if indicator:
        df_temp = df_temp[df_temp['Indicator']==indicator]

    

    new_data = [transform(record) for record in df_temp[:1000].to_dict(orient='records')]
    return new_data

@app.get("/group/indicator",response_model=List[ClimategroupIndicator] ,tags=["Indicator"])
async def getIndicatorResponse(
    country:str = Query(None, description="Filter by Country"),
    indicator : str = Query(None, description="Filter by Indicator")):

    """Retrieve all Indicator response records"""
    df_temp = df.groupby(['Country','Sector','Indicator','Unit'])[['2005',
       '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
       '2015', '2016', '2017', '2018', '2022']].sum().reset_index()


    if country:
        df_temp = df_temp[df_temp['Country']==country]

    if indicator:
        df_temp = df_temp[df_temp['Indicator']==indicator]

    

    new_data = [transform(record) for record in df_temp[:1000].to_dict(orient='records')]
    return new_data


@app.get("/country",response_model=List[ClimateCountry] ,tags=["Country"])
async def getIsoResponse(
    country:str = Query(None, description="Filter by country"),
    ):

    """Retrieve all Indicator response records"""
    df_temp = df.groupby(['Country','ISO3'])[['2005',
       '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
       '2015', '2016', '2017', '2018', '2022']].sum().reset_index()


    if country:
        df_temp = df_temp[df_temp['Country']==country]

    new_data = [transform(record) for record in df_temp[:1000].to_dict(orient='records')]
    return new_data    

@app.get("/chat",tags=["Chatbot"])
async def getchatResponse(
    query:str = Query(None, description="user query"),
    ):
    
    
    res = model.invoke(prompt_template.invoke({"query": query}))

    return res.content
    

if __name__ == "__main__":
    uvicorn.run("api:app", port=5000, log_level="info",reload=True)
