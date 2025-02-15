import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Hashable, Optional

from sqlalchemy import (
    select,
    func,
    and_,
    PrimaryKeyConstraint,
)
from sqlalchemy.dialects.postgresql import insert

from loaders.base import DatabaseDataManager
from database.models.predictions import PredictionsTable
from database.schemas.predictions import PredictionsSchema


@dataclass
class PredictionsDataManager(DatabaseDataManager):
    """
    Loads the data from the predictions to the database.
    """
    
    def __post_init__(self) -> None:
        super().__post_init__()
        self.model_table = PredictionsTable
        self.model_schema = PredictionsSchema

    def load(self, 
            start_date: datetime, 
            end_date: datetime,
            source_name: str,
            location_code: str,
            target_variables: Optional[list[str]] = None,
            filter_columns: Optional[list[str]] = None
        ) -> pd.DataFrame:
        conditions_list = ['None', None, '', False]
        conditions = [
            self.model_table.source_name == source_name,
            self.model_table.location_code == location_code,
        ]

        # Add the date to the conditions if it is not in conditions_list
        if start_date not in conditions_list and end_date not in conditions_list:
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
                end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
            conditions.append(self.model_table.timestamp.between(start_date, end_date))

        # Add condition for variable codes
        if target_variables not in conditions_list:
            # Ensure target_variables is a list or tuple
            if not isinstance(target_variables, (list, tuple)):
                target_variables = [target_variables]
            conditions.append(self.model_table.target_variable.in_(target_variables))

        # Add the last date to the conditions if it is not in conditions_list
        if start_date in conditions_list or end_date in conditions_list:
            conditions.append(self.model_table.timestamp.op('=')(
                select(func.max(self.model_table.timestamp)).where(
                    and_(*conditions))))

        # Create the statement with the combined conditions
        statement = select(self.model_table).where(and_(*conditions)
        ).order_by(self.model_table.timestamp.desc())

        # Execute the query, get the results and close the session
        with self.Session.begin() as session:
            # Assigns value to query if it is not None
            if query := session.execute(statement).all():
                data = [self.model_schema(
                    timestamp=row[0].timestamp,
                    model_date=row[0].model_date,
                    model_name=row[0].model_name,
                    model_type=row[0].model_type,
                    model_version=row[0].model_version,
                    latitude=row[0].latitude,
                    longitude=row[0].longitude,
                    target_variable=row[0].target_variable,
                    real_value=row[0].real_value,
                    predictions=row[0].predictions,
                    data_quality_percentage=row[0].data_quality_percentage,
                    validation_boolean=row[0].validation_boolean,
                    suggested_value=row[0].suggested_value,
                    update_date=row[0].update_date,
                ).model_dump() for row in query] # type: ignore

                df = pd.DataFrame(data)

                # Add the filter columns to the statement if they are not in
                # conditions_list
                if filter_columns not in conditions_list:
                    # Transform the columns_list list into a regex string
                    filter_columns_str = '|'.join(filter_columns)

                    # Filter the columns
                    df = df.filter(regex=filter_columns_str)

                return df
        
    def _statement_insert_to_db(self, data: list[dict[Hashable, Any]]) -> None:
        """
        Inserts the data from the dataframe to the database.
        
        Parameters:
        -----------
        data: dict
            Dictionary with the data to insert into the database.

        Returns:
        --------
        None
            Data inserted into the database.
        """
        try:
            statement = insert(self.model_table).values(data)
            statement = statement.on_conflict_do_update(
                constraint=PrimaryKeyConstraint(
                    self.model_table.timestamp,
                    self.model_table.latitude,
                    self.model_table.longitude,
                    self.model_table.target_variable
                ),
                set_=dict(
                    model_date=statement.excluded.model_date,
                    model_name=statement.excluded.model_name,
                    model_type=statement.excluded.model_type,
                    model_version=statement.excluded.model_version,
                    real_value=statement.excluded.real_value,
                    predictions=statement.excluded.predictions,
                    data_quality_percentage=statement.excluded.data_quality_percentage,
                    validation_boolean=statement.excluded.validation_boolean,
                    suggested_value=statement.excluded.suggested_value,
                    update_date=datetime.now(),
                )
            )

            # Execute the query, commit and close the session
            with self.Session.begin() as session:
                session.execute(statement)

        except Exception as e:
            self.logger.debug(e)
            raise e