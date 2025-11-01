# %%
import pandas as pd
from numpy.ma.core import absolute

import parameters
from scipy.stats import poisson

rate_table = pd.read_csv("data/rate_table.csv", header=0)


def flood_frequency(height, mu, scale, shape):
    number_exceeds_height = (1 + (shape * (height - mu)) / scale) ** (-1 / shape)
    # mu = mu + mu_{SLR}
    return number_exceeds_height


# def SLR(projection_year, scenario):
#     location = 0.0017 * projection_year + scenario * projection_year**2
#     return location


def get_rate_NFIP(zone, height):
    # print(zone, height)
    # TODO: make sure this is correct
    building_rate = (
        rate_table.loc[
            (rate_table["zone"] == zone) & (rate_table["height"] <= height), "building"
        ]
        .sort_values(ascending=True)
        .values[0]
    )
    contents_rate = (
        rate_table.loc[
            (rate_table["zone"] == zone) & (rate_table["height"] <= height), "contents"
        ]
        .sort_values(ascending=True)
        .values[0]
    )
    # print(zone, height)
    # print(building_rate)
    return building_rate, contents_rate


def insurance_rate(
        insurance_type,
        zone,
        height,
        CRS_rewards,
):
    if insurance_type == "NFIP":
        building_rate, contents_rate = get_rate_NFIP(zone, height)

        building_rate -= building_rate * CRS_rewards
        contents_rate -= contents_rate * CRS_rewards

        # rate = building_rate + contents_rate

    if insurance_type == "private":
        # “we build our private insurance model based on a linear relationship between the insurance rates and the freeboard height of the house relative to the BFE and with rates that are three times higher than NFIP rates for high-risk properties” ([Han and Peng, 2019, p. 73] #TODO
        building_rate, contents_rate = [
            rate * 3 for rate in insurance_rate("NFIP", zone, height, CRS_rewards)
        ]

    return building_rate, contents_rate


def elevation_cost(elevation, area):
    if elevation <= 0:
        cost = 0
    if elevation <= 2 and elevation > 0:
        cost = 17 * area
    if elevation > 2:
        cost = (17 + (elevation - 2) * 0.75) * area
    return cost


def damage_assessment(building_type, flood_elevation_feet):
    flood_elevation = 0.3048 * flood_elevation_feet
    if building_type == "residential":
        damage = (
                         0.2391 * (flood_elevation ** 3)
                         - 3.5524 * (flood_elevation ** 2)
                         + 19.933 * flood_elevation
                         + 11.623
                 ) / 100
    else:
        damage = (
                         -0.1347 * (flood_elevation ** 3)
                         + 1.1448 * (flood_elevation ** 2)
                         + 9.1078 * flood_elevation
                         + 4.4057
                 ) / 100
    return damage


def risk_mitigation_cost(
        self,
        house_elevation,
        insurance_coverage,
        area,
):
    if house_elevation <= 2:
        cost = 17 * area * house_elevation
    else:
        cost = 17 * area * 2 + 0.75 * area * (house_elevation - 2)

    return cost


# def decide_adaptation(self, expected_annual_risk):
#     adaptation_decision = 1  # TODO
#     return adaptation_decision


def risk_perception(income, race, education, ownership, government):
    risk_perception = (
                              parameters.a * income
                              + parameters.b * race
                              + parameters.c * education
                              + parameters.d * ownership
                              + parameters.e * government
                      ) / (parameters.a + parameters.b + parameters.c + parameters.d + parameters.e)
    return risk_perception


def pi_calculation(risk_perception, return_period):
    # Calculate the probability of one event happening
    # probability = poisson.pmf(1, 1 / return_period)
    probability = 1 / return_period
    # print(return_period, 1 / return_period, poisson.pmf(1, 1 / return_period))
    # Calculate 'core' ensuring it's non-negative
    core = min(1, 10 ** (2 * risk_perception - 1) * probability)
    # print('return_period', return_period)
    # print('risk_perception', risk_perception)
    # print('probability', probability)
    # print('core', core)

    # Calculate the numerator and denominator separately
    numerator = core ** parameters.gamma
    denominator = (numerator + ((1 - core) ** parameters.gamma)) ** (
            1 / parameters.gamma
    )
    # print("numerator, denominator", numerator, denominator)
    # Check denominator to avoid division by zero

    # Return the result as a float
    return float(numerator / denominator)


def U(damage):
    return damage ** parameters.expected_utility_parameter


def prospect_utility_action(
        flood_elevation_list,
        return_period_list,
        house_elevation,
        building_type,
        risk_perception,
        total_annual_cost,
        house_value,
        insurance_coverage,
        public_risk_reduction,
        only_EAD,
):
    PU = 0
    EAD = 0
    damage_list = []
    for i in range(len(flood_elevation_list)):
        flood_height = max(0, float(flood_elevation_list[i]) - house_elevation)
        # if flood_height > 0:
        #     print('flood_height', flood_height)
        damage_percentage = damage_assessment(building_type, flood_height)
        # if damage_percentage - public_risk_reduction < 0:
        #     damage = 0
        # else:
        #     damage = house_value * (damage_percentage - public_risk_reduction)
        damage = house_value * damage_percentage * (1 - public_risk_reduction)
        damage_list.append(damage)

        # print("flood_height, damage_percentage")
        # print(flood_height, damage_percentage)

        if only_EAD == False:
            pi = pi_calculation(risk_perception, return_period_list[i])

            # print(
            #     "pi, U, PU", pi, U(damage + total_annual_cost - insurance_coverage), PU
            # )
            # print('U_value', damage + total_annual_cost - insurance_coverage)
            total_damage = max(0, damage - insurance_coverage) + total_annual_cost
            # absolute_total_damage = absolute(total_damage)
            PU += pi * U(total_damage)

    # print("damage_list", damage_list)
    if len(flood_elevation_list) > 1:
        # print(flood_elevation_list)
        for i in range(len(flood_elevation_list) - 1):
            EAD += (
                    1
                    / 2
                    * (damage_list[i] + damage_list[i + 1])
                    * (1 / return_period_list[i] - 1 / return_period_list[i + 1])
            )
    else:
        EAD = damage_list[0] / return_period_list[0]

    if only_EAD == True:
        return EAD, damage_list
    else:
        return PU, EAD, damage_list
