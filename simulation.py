import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from model import adaptation_simulation

structure_df = pd.read_csv("data/full_data_for_simulation.csv", header=0)

structure_df = structure_df[(structure_df["education"].notna()) & (structure_df["property_flood_zone"] != 'OPEN')]
# print(structure_df["flood_elevation_list"])

census_tract_number_list = [0, 10, 25, 50]
# census_tract_number_list = [0]
policy_list = ['pre_FIRM', 'voucher']

for census_tract_number in census_tract_number_list:
    for policy_name in policy_list:
        simulation_model = adaptation_simulation(
            # structure_dataframe=structure_df.head(5),
            structure_dataframe=structure_df,
            return_period_list=[5.886, 13.734, 24.7212, 61.803, 200],
            policy=policy_name,
            CRS_rewards=0.25,
            covered_census_tracts=census_tract_number,
            risk_reduction_percentage=0.25,
        )

        simulation_model.agent_generation()

        # print(simulation_model.structure_dataframe)

        simulation_model.step()

        result = simulation_model.datacollector.get_agent_vars_dataframe()
        result.reset_index(inplace=True)
        result_df = structure_df.merge(
            result, left_on="structure_id", right_on="AgentID", how="left"
        )
        result_df.to_csv("data/result_{}_{}.csv".format(policy_name, census_tract_number), index=False)
