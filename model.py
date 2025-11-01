# %%
import matplotlib.pyplot as plt
import pandas as pd
import mesa
import agent
import collections
from mesa.space import ContinuousSpace
from mesa.time import SimultaneousActivation
from mesa.agent import AgentSet
import functions
import numpy as np
from ast import literal_eval


class adaptation_simulation(mesa.Model):
    def __init__(
            self,
            structure_dataframe,
            return_period_list: list,
            policy,  # voucher or pre_FIRM
            CRS_rewards,  # 25% or 45%(50%)
            covered_census_tracts,  # 100 or 500
            risk_reduction_percentage,
    ):
        super().__init__()

        self.structure_dataframe = structure_dataframe

        self.return_period_list = return_period_list

        self.policy = policy
        self.CRS_rewards = CRS_rewards
        self.covered_census_tracts = covered_census_tracts
        self.risk_reduction_percentage = risk_reduction_percentage

        self.elevation_options = [0, 2, 4, 6, 8]
        self.NFIP_coverage_options = [60000, 150000, 250000]
        self.private_coverage_options = [60000, 250000, 500000]

        self.schedule = SimultaneousActivation(self)

        self.datacollector = mesa.DataCollector(
            agent_reporters={'EAD': 'EAD',
                             'insurance_type': 'insurance_type',
                             'insurance_coverage': 'insurance_coverage',
                             'elevation': 'elevation',
                             'damage_list': 'damage_list',
                             }
        )

    def agent_generation(self):
        self.structure_dataframe['initial_EAD'] = float('nan')
        self.structure_dataframe['public_risk_reduction'] = float('nan')
        for idx, row in self.structure_dataframe.iterrows():
            EAD, damage_list = functions.prospect_utility_action(
                literal_eval(row['flood_elevation_list']),
                self.return_period_list,
                row['property_height'],
                row['building_type'],
                0,
                0,
                row['house_value'],
                0,
                0,
                True,
            )
            self.structure_dataframe.at[idx, 'initial_EAD'] = EAD

        # Sort the dataframe by initial_EAD
        self.census_dataframe = self.structure_dataframe.groupby('GEOID', as_index=False).agg(
            {'initial_EAD': 'mean'}).sort_values(by='initial_EAD', ascending=False)

        # self.structure_dataframe = self.structure_dataframe.sort_values(by='initial_EAD', ascending=False)
        self.max_initial_EAD = self.census_dataframe['initial_EAD'][0]

        self.top_census_tracts = self.census_dataframe.head(self.covered_census_tracts)['GEOID'].to_list()

        self.structure_dataframe['public_risk_reduction'] = 0.0
        for idx, row in self.structure_dataframe.iterrows():
            if row['GEOID'] in self.top_census_tracts:
                self.structure_dataframe.at[idx, 'public_risk_reduction'] = self.risk_reduction_percentage
        # print(self.structure_dataframe[self.structure_dataframe['public_risk_reduction'] > 0]['GEOID'])
        # self.structure_dataframe.iloc[:self.covered_census_tracts,
        # self.structure_dataframe.columns.get_loc('public_risk_reduction')] = self.risk_reduction_percentage

        for idx, row in self.structure_dataframe.iterrows():
            household_agent = agent.household(
                unique_id=row["structure_id"],
                model=self,
                mortgage=row["mortgage"],
                income=row["income"],
                race=row["race"],
                education=row["education"],
                ownership=row['ownership'],
                flood_elevation_list=literal_eval(row['flood_elevation_list']),
                property_flood_zone=row['property_flood_zone'],
                property_height=row['property_height'],
                area=row['area'],
                building_type=row['building_type'],
                house_value=row['house_value'],
                BFE=row['BFE'],
                initial_EAD=EAD,
                public_risk_reduction=row['public_risk_reduction'],
            )
            self.schedule.add(household_agent)
        self.household_id = self.structure_dataframe["structure_id"].tolist()

    def storm_surge(self):
        self.storm_surge_height = 0
        return self.storm_surge_height

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
