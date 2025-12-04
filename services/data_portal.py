from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, MutableMapping, Optional

import httpx
from fastapi import HTTPException
from pydantic import AnyHttpUrl


@dataclass(frozen=True)
class DatasetConfig:
    """Metadata describing a supported remote dataset."""

    key: str
    name: str
    description: str
    source_url: AnyHttpUrl
    default_params: Mapping[str, str] = field(default_factory=dict)


class ClimateDataPortalClient:
    """Client for the IMF climate data portal and connected ArcGIS services."""

    def __init__(self, datasets: List[DatasetConfig]) -> None:
        self.datasets: Dict[str, DatasetConfig] = {dataset.key: dataset for dataset in datasets}

    def list_datasets(self) -> List[DatasetConfig]:
        """Return the configured datasets."""

        return list(self.datasets.values())

    def get_dataset(self, key: str) -> DatasetConfig:
        """Retrieve dataset configuration or raise a 404."""

        dataset = self.datasets.get(key)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset '{key}' is not supported")
        return dataset

    async def fetch_dataset(
        self,
        dataset_key: str,
        where: Optional[str] = None,
        out_fields: Optional[str] = None,
        out_sr: Optional[int] = None,
        limit: Optional[int] = None,
        extra_params: Optional[MutableMapping[str, str]] = None,
    ) -> MutableMapping[str, object]:
        """
        Query a dataset from the portal.

        Parameters mirror the ArcGIS feature service API for flexibility.
        """

        dataset = self.get_dataset(dataset_key)
        params: MutableMapping[str, object] = {**dataset.default_params}

        params["where"] = where or dataset.default_params.get("where", "1=1")
        params["outFields"] = out_fields or dataset.default_params.get("outFields", "*")
        params["f"] = dataset.default_params.get("f", "json")

        if out_sr:
            params["outSR"] = out_sr
        if limit:
            params["resultRecordCount"] = limit
        if extra_params:
            params.update(extra_params)

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(str(dataset.source_url), params=params)

        try:
            response.raise_for_status()
            payload: MutableMapping[str, object] = response.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=502, detail="Failed to parse dataset response") from exc

        if isinstance(payload, Mapping) and payload.get("error"):
            message = payload.get("error", {}).get("message", "Unknown error from dataset")
            raise HTTPException(status_code=502, detail=message)

        return payload


REMOTE_DATASETS: List[DatasetConfig] = [
    DatasetConfig(
        key="ghg_emissions_quarterly",
        name="Quarterly Greenhouse Gas (GHG) Air Emissions Accounts",
        description=(
            "Quarterly GHG air emissions accounts sourced from the IMF climate data portal "
            "and hosted on ArcGIS feature services."
        ),
        source_url=AnyHttpUrl(
            "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
            "Indicator_1_1_quarterly/FeatureServer/0/query"
        ),
        default_params={"outSR": "4326", "f": "json", "where": "1=1", "outFields": "*"},
    ),
    DatasetConfig(
        key="ghg_emissions_annual",
        name="Annual Greenhouse Gas (GHG) Air Emissions Accounts",
        description="Annual GHG emissions accounts published on the IMF climate data portal.",
        source_url=AnyHttpUrl(
            "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
            "Indicator_1_1_annual/FeatureServer/0/query"
        ),
        default_params={"outSR": "4326", "f": "json", "where": "1=1", "outFields": "*"},
    ),
]

client = ClimateDataPortalClient(REMOTE_DATASETS)
