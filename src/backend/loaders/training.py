import pickle
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Hashable, Optional

from sqlalchemy import (
    select,
    func,
    and_,
    delete,
    PrimaryKeyConstraint,
)
from sqlalchemy.dialects.postgresql import insert

from loaders.base import DatabaseDataManager
from database.models.training import (
    ModelVersioningTable,
    TrainingConfigurationTable,
)
from database.schemas.training import (
    ModelVersioningSchema,
    TrainingConfigurationSchema,
)


@dataclass
class ModelVersioningManager(DatabaseDataManager):
    def load(self):
        pass

    def get_model_version(self,
            model_type: str,
            latitude: float,
            longitude: float,
            target_variable: str
        ) -> int:
        statement = select(ModelVersioningTable).where(
            ModelVersioningTable.model_type == model_type,
            ModelVersioningTable.latitude == str(latitude),
            ModelVersioningTable.longitude == str(longitude),
            ModelVersioningTable.target_variable == target_variable
        )
        with self.Session.begin() as session:
            if query := session.execute(statement).all():
                return max(row[0].model_version for row in query) + 1
            return 1
    
    def get_recent_models_from_db(self,
            model_type: str,
            latitude: float,
            longitude: float,
            target_variable: Optional[str] = None,
            columns: list[str] = []
        ) -> list[Any]:
        conditions_list = ['None', None, '']
        conditions = [
            and_(
                ModelVersioningTable.model_type == model_type,
                ModelVersioningTable.latitude == str(latitude),
                ModelVersioningTable.longitude == str(longitude),
            )
        ]

        if target_variable not in conditions_list:
            conditions.append(ModelVersioningTable.target_variable.op('~')(f'{target_variable}'))

        statement = select(
            *[getattr(ModelVersioningTable, col) for col in columns]
        ).where(and_(*conditions)).order_by(
            ModelVersioningTable.target_variable,
            ModelVersioningTable.model_version.desc(),
            ModelVersioningTable.model_date.desc()
        ).distinct(ModelVersioningTable.target_variable)

        with self.Session.begin() as session:
            if query := session.execute(statement).all():
                return list(query) # type: ignore
        return []
            
    def get_best_model_from_date_range(self,
            model_type: str,
            latitude: float,
            longitude: float,
            target_variable: Optional[str] = None,
            columns: list[str] = []
        ) -> list[Any]:
        conditions = [
            ModelVersioningTable.model_type == model_type,
            ModelVersioningTable.latitude == str(latitude),
            ModelVersioningTable.longitude == str(longitude),
        ]

        if target_variable:
            conditions.append(ModelVersioningTable.target_variable == target_variable)

        latest_date_query = select(
            func.max(ModelVersioningTable.model_date)
        ).where(
            and_(*conditions)
        ).scalar_subquery()

        date_range_conditions = [
            ModelVersioningTable.model_date.between(
                latest_date_query - timedelta(days=1),
                latest_date_query + timedelta(days=1)
            )
        ]

        max_accuracy_subquery = select(
            func.max(ModelVersioningTable.model_forecast_accuracy)
        ).where(
            and_(*conditions, *date_range_conditions)
        ).scalar_subquery()

        accuracy_conditions = [
            ModelVersioningTable.model_forecast_accuracy == max_accuracy_subquery
        ]

        if columns:
            statement = select(
                *[getattr(ModelVersioningTable, col) for col in columns]
            ).where(
                and_(*conditions, *date_range_conditions, *accuracy_conditions)
            ).order_by(
                ModelVersioningTable.model_version.desc(),
                ModelVersioningTable.model_date.desc()
            )
        else:
            statement = select(ModelVersioningTable).where(
                and_(*conditions, *date_range_conditions, *accuracy_conditions)
            ).order_by(
                ModelVersioningTable.model_version.desc(),
                ModelVersioningTable.model_date.desc()
            )

        with self.Session.begin() as session:
            if query := session.execute(statement).all():
                return list(query)  # type: ignore
        return []

    def load_regression_models(self,
            latitude: float,
            longitude: float,
            target_variable: str,
            model_type: str,
        ) -> dict[str, tuple]:
        regression_models = self.get_best_model_from_date_range(
            model_type=model_type,
            latitude=latitude,
            longitude=longitude,
            target_variable=target_variable,
            columns=[
                'model_date',
                'model_name',
                'model_type',
                'latitude',  
                'longitude',
                'target_variable',
                'model_feature_names',
                'model_version',
                'model_metric_test_value',
                'model_forecast_accuracy',
                'standard_scaler_binary',
                'polynomial_transformer_binary',
                'model_binary',
                'feature_importance_variables',
                'feature_importance_values'
            ]
        )
        return {
            target_variable: (
                model_date,
                model_name,
                model_type,
                latitude,
                longitude,
                model_feature_names,
                model_version,
                model_metric_test_value,
                model_forecast_accuracy,
                pickle.loads(standard_scaler_binary) if standard_scaler_binary is not None else None,
                pickle.loads(polynomial_transformer_binary) if polynomial_transformer_binary is not None else None,
                pickle.loads(model_binary) if model_binary is not None else None,
                feature_importance_variables,
                feature_importance_values
            ) for model_date,
                model_name,
                model_type,
                latitude,
                longitude,
                target_variable,
                model_feature_names,
                model_version,
                model_metric_test_value,
                model_forecast_accuracy,
                standard_scaler_binary,
                polynomial_transformer_binary,
                model_binary,
                feature_importance_variables,
                feature_importance_values in regression_models
        }

    def get_selected_models(self,
            model_date: datetime,
            model_name: str,
            model_type: str,
            latitude: float,
            longitude: float,
            target_variable: str,
            columns: list[str] = []
        ) -> list[Any]:
        conditions_list = ['None', None, '']
        conditions = [
            and_(
                ModelVersioningTable.model_name == model_name,
                ModelVersioningTable.model_type == model_type,
                ModelVersioningTable.latitude == str(latitude),
                ModelVersioningTable.longitude == str(longitude),
                ModelVersioningTable.target_variable.op('~')(f'{target_variable}')
            )
        ]

        if model_date not in conditions_list:
            start_date = model_date - timedelta(seconds=1)
            end_date = model_date + timedelta(seconds=1)
            conditions.append(
                and_(
                    ModelVersioningTable.model_date >= start_date,
                    ModelVersioningTable.model_date <= end_date
                )
            )

        statement = select(
            *[getattr(ModelVersioningTable, col) for col in columns]
        ).where(and_(*conditions)).order_by(
            ModelVersioningTable.target_variable,
            ModelVersioningTable.model_version,
            ModelVersioningTable.model_date
        )

        with self.Session.begin() as session:
            if query := session.execute(statement).all():
                return list(query) # type: ignore
        return []

    def get_models_for_weights_visualization(self,
            latitude: float,
            longitude: float,
            model_name: str,
            model_type: str,
            target_variable: str,
            model_date: datetime
        ) -> dict[str, tuple]:
    
        regression_models = self.get_selected_models(
            model_name=model_name,
            model_date=model_date,
            model_type=model_type,
            latitude=latitude,
            longitude=longitude,
            target_variable=target_variable,
            columns=[
                'model_date',
                'model_name',
                'model_type',
                'latitude',
                'longitude',
                'target_variable',
                'model_feature_names',
                'model_version',
                'model_metric_test_value',
                'model_forecast_accuracy',
                'feature_importance_variables',
                'feature_importance_values'
            ]
        )
        
        return {
            target_variable: (
                model_date,
                model_name,
                model_type,
                latitude,
                longitude,
                model_feature_names,
                model_version,
                model_metric_test_value,
                model_forecast_accuracy,
                feature_importance_variables,
                feature_importance_values
            ) for model_date,
                model_name,
                model_type,
                latitude,
                longitude,
                target_variable,
                model_feature_names,
                model_version,
                model_metric_test_value,
                model_forecast_accuracy,
                feature_importance_variables,
                feature_importance_values in regression_models
        }

    def _statement_insert_to_db(self, data: list[dict[Hashable, Any]]) -> None:
        try:
            statement = insert(ModelVersioningTable).values(data)
            statement = statement.on_conflict_do_update(
                constraint=PrimaryKeyConstraint(
                    ModelVersioningTable.model_date,
                    ModelVersioningTable.model_name,
                    ModelVersioningTable.model_type,
                    ModelVersioningTable.latitude,
                    ModelVersioningTable.longitude,
                    ModelVersioningTable.target_variable,
                ),
                set_=dict(
                    model_feature_names=statement.excluded.model_feature_names,
                    model_features_count=statement.excluded.model_features_count,
                    model_version=statement.excluded.model_version,
                    model_parameters=statement.excluded.model_parameters,
                    model_metric_name=statement.excluded.model_metric_name,
                    model_metric_validation_value=statement.excluded.model_metric_validation_value,
                    model_metric_test_value=statement.excluded.model_metric_test_value,
                    model_forecast_accuracy=statement.excluded.model_forecast_accuracy,
                    standard_scaler_binary=statement.excluded.standard_scaler_binary,
                    polynomial_transformer_binary=statement.excluded.polynomial_transformer_binary,
                    model_binary=statement.excluded.model_binary,
                    feature_importance_variables=statement.excluded.feature_importance_variables,
                    feature_importance_values=statement.excluded.feature_importance_values
                )
            )

            with self.Session.begin() as session:
                session.execute(statement)

        except Exception as e:
            self.logger.debug(e)
            raise e


@dataclass
class TrainingConfigurationManager(DatabaseDataManager):

    def load(self, date: str) -> Optional[pd.DataFrame]:
        conditions = [
            TrainingConfigurationTable.timestamp == date
        ]

        statement = select(TrainingConfigurationTable).where(and_(*conditions)
        ).order_by(TrainingConfigurationTable.created_date.asc())

        with self.Session.begin() as session:
            if query := session.execute(statement).all():
                results = [TrainingConfigurationSchema(
                    timestamp=row[0].timestamp,
                    stage=row[0].stage,
                    parameter_code=row[0].parameter_code,
                    parameter_value=row[0].parameter_value,
                    created_date=row[0].created_date,
                    update_date=row[0].update_date
                ).model_dump() for row in query] # type: ignore

                return pd.DataFrame(results)

    def _statement_insert_to_db(self, data: list[dict[Hashable, Any]]) -> None:
        try:
            statement = insert(TrainingConfigurationTable).values(data)
            statement = statement.on_conflict_do_update(
                constraint=PrimaryKeyConstraint(
                    TrainingConfigurationTable.timestamp,
                    TrainingConfigurationTable.stage,
                    TrainingConfigurationTable.parameter_code,
                ),
                set_=dict(
                    model_id=statement.excluded.model_id,
                    parameter_value=statement.excluded.parameter_value,
                    update_date=statement.excluded.update_date
                )
            )
            with self.Session.begin() as session:
                session.execute(statement)
        except Exception as e:
            self.logger.debug(e)
            raise e

    def delete(self, date: str) -> None:

        conditions = [TrainingConfigurationTable.timestamp == date]

        try:
            statement = delete(TrainingConfigurationTable).where(and_(*conditions))

            with self.Session.begin() as session:
                session.execute(statement)

        except Exception as e:
            self.logger.debug(e)
            raise e