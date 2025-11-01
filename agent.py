from mesa import Agent
import pandas as pd
import parameters
import functions
import numpy as np
import random
import numpy_financial as npf

rate_table = pd.read_csv("data/rate_table.csv", header=0)


class household(Agent):
    def __init__(
            self,
            unique_id,
            model,
            mortgage,
            # socioeconomic: list,  # including agent income, property ownership, education level, occupied property value, and social connections, etc
            income,
            race,
            education,
            ownership,
            # insurance_cost,
            # insurance_coverage,
            flood_elevation_list: list,
            property_flood_zone,
            property_height,
            area,
            building_type,
            house_value,
            BFE,
            initial_EAD,
            public_risk_reduction,
    ):
        super().__init__(unique_id, model)

        self.mortgage = mortgage

        self.income = income
        self.race = race
        self.eduction = int(education)
        self.ownership = ownership

        # self.insurance_cost = insurance_cost
        # self.insurance_coverage = insurance_coverage

        self.flood_elevation_list = flood_elevation_list

        self.property_flood_zone = property_flood_zone
        self.property_height = property_height
        self.area = area
        self.house_value = house_value

        # self.property_year = 0
        # self.property_year_max = 30

        # self.insurance_premium_rate = 0
        # self.hard_adaptive_measure = 0
        # self.hard_adaptive_measure_year = 0
        # self.hard_adaptive_measure_year_max = 30

        self.property_ownership = 0

        self.building_type = building_type

        self.BFE = BFE
        self.initial_EAD = initial_EAD

        self.PU = parameters.M

        self.public_risk_reduction = public_risk_reduction

        # “In our study, we randomly selected 34% of households in flood A zone and V zone as households with mortgages, who are required to have flood insurance”
        if self.property_flood_zone in ["A", "VE", "VO"] and self.mortgage == "Housing Units with a Mortgage":
            # TODO: double check A and V zones
            self.require_insurance = True
        else:
            self.require_insurance = False
            # if random.uniform(0, 1) < 0.34:
            #     self.require_insurance = True
            # else:
            #     self.require_insurance = False

        income_mapping = {
            "Income Below $45,000": 0,
            "Households with Income $45,000 - $49,999": (45000 - 45000)
                                                        / (200000 - 45000),
            "Households with Income $50,000 - $59,999": (50000 - 45000)
                                                        / (200000 - 45000),
            "Households with Income $60,000 - $74,999": (60000 - 45000)
                                                        / (200000 - 45000),
            "Households with Income $75,000 - $99,999": (75000 - 45000)
                                                        / (200000 - 45000),
            "Households with Income $100,000 - $124,999": (100000 - 45000)
                                                          / (200000 - 45000),
            "Households with Income $125,000 - $149,999": (125000 - 45000)
                                                          / (200000 - 45000),
            "Households with Income $150,000 - $199,999": (150000 - 45000)
                                                          / (200000 - 45000),
            "Households with Income $200,000 or more": 1,
        }

        self.i_income = income_mapping.get(self.income, 0)

        if self.race == "Minority Population":
            self.i_race = 1
        else:
            self.i_race = 0

        if self.ownership == "Owner-Occupied Housing Units":
            self.i_ownership = 1
        else:
            self.i_ownership = 0

        # education_to_quantitative = {
        #     "No schooling completed": 1,
        #     "Nursery school": 2,
        #     "Kindergarten": 3,
        #     "1st Grade": 4,
        #     "2nd Grade": 5,
        #     "3rd Grade": 6,
        #     "4th Grade": 7,
        #     "5th Grade": 8,
        #     "6th Grade": 9,
        #     "7th Grade": 10,
        #     "8th Grade": 11,
        #     "9th Grade": 12,
        #     "10th Grade": 13,
        #     "11th Grade": 14,
        #     "12th Grade, no diploma": 15,
        #     "Regular high school diploma": 16,
        #     "GED or alternative credential": 17,
        #     "Some college, less than 1 year": 18,
        #     "Some college, 1 or more years, no degree": 19,
        #     "Associate degree": 20,
        #     "Bachelor Degree": 21,
        #     "Master Degree": 22,
        #     "Professional School Degree": 23,
        #     "Doctorate Degree": 24,
        # }

        max_education_level = 24
        self.i_eduction = self.eduction / max_education_level

        if self.public_risk_reduction == 0:
            self.i_government = 0.5
        else:
            self.i_government = (
                    0.5 + self.initial_EAD / self.model.max_initial_EAD * 1 / 2
            )

    def step(self):
        self.risk_perception = functions.risk_perception(
            self.i_income,
            self.i_race,
            self.i_eduction,
            self.i_ownership,
            self.i_government,
        )

        # print(self.risk_perception)

        PU_no_action, EAD, damage_list = functions.prospect_utility_action(
            self.flood_elevation_list,
            self.model.return_period_list,
            self.property_height,
            self.building_type,
            self.risk_perception,
            0,
            self.house_value,
            0,
            self.public_risk_reduction,
            False,
        )

        self.PU_no_action = PU_no_action
        self.EAD_no_action = EAD
        self.insurance_type = 'No insurance'
        self.insurance_coverage = 0
        self.damage_list = damage_list

        print("PU_no_action", self.PU_no_action)
        print('EAD_no_action', self.EAD_no_action)

        if self.model.policy == "voucher":
            self.interest_rate = 0.03
            self.loan_length = 30

            estimated_elevation = max(0, self.BFE - self.property_height + 1)

            self.elevation = estimated_elevation
            print("elevation", self.elevation)

            elevation_cost = functions.elevation_cost(estimated_elevation, self.area)
            # print("elevation_cost", elevation_cost)

            annual_elevation_cost = -npf.pmt(
                self.interest_rate, self.loan_length, elevation_cost
            )
            # print("annual_elevation_cost", annual_elevation_cost)

            building_rate, contents_rate = functions.insurance_rate(
                "NFIP",
                self.property_flood_zone,
                estimated_elevation + self.property_height - self.BFE,
                self.model.CRS_rewards,
            )

            for coverage in self.model.NFIP_coverage_options:

                insurance_cost = coverage / 100 * building_rate

                total_annual_cost = insurance_cost + annual_elevation_cost

                income_mapping = {
                    "Income Below $45,000": 45000,
                    "Households with Income $45,000 - $49,999": 47500,
                    "Households with Income $50,000 - $59,999": 55000,
                    "Households with Income $60,000 - $74,999": 67500,
                    "Households with Income $75,000 - $99,999": 87500,
                    "Households with Income $100,000 - $124,999": 112500,
                    "Households with Income $125,000 - $149,999": 137500,
                    "Households with Income $150,000 - $199,999": 175000,
                    "Households with Income $200,000 or more": 200000,
                }

                income_value = income_mapping.get(self.income, 45000)  # Default to the lowest income if not found

                if total_annual_cost > income_value / 12 * 0.05:
                    total_annual_cost = income_value / 12 * 0.05

                PU, EAD, damage_list = functions.prospect_utility_action(
                    self.flood_elevation_list,
                    self.model.return_period_list,
                    self.property_height + self.elevation,
                    self.building_type,
                    self.risk_perception,
                    total_annual_cost,
                    self.house_value,
                    coverage,
                    self.public_risk_reduction,
                    False,
                )

                # print("PU", PU)
                #
                # print('EAD', EAD)

                if PU < self.PU:
                    self.PU = PU
                    self.insurance_type = "NFIP"
                    self.insurance_coverage = coverage
                    self.EAD = EAD
                    self.damage_list = damage_list
                #
                # print('self.PU', self.PU)
                # print('self.EAD', self.EAD)
                # print(" ")


        else:
            self.interest_rate = 0.04
            self.loan_length = 20

            for elevation in self.model.elevation_options:
                elevation_cost = functions.elevation_cost(elevation, self.area)

                annual_elevation_cost = -npf.pmt(
                    self.interest_rate, self.loan_length, elevation_cost
                )

                print('annual_elevation_cost', annual_elevation_cost)

                self.new_property_height = self.property_height + elevation

                # if self.require_insurance == False:
                #     PU, EAD = functions.prospect_utility_action(
                #         self.flood_elevation_list,
                #         self.model.return_period_list,
                #         self.new_property_height,
                #         self.building_type,
                #         self.risk_perception,
                #         annual_elevation_cost,
                #         self.house_value,
                #         0,
                #         self.public_risk_reduction,
                #         False,
                #     )
                #
                #     if PU < self.PU:
                #         self.PU = PU
                #         self.insurance_type = "No insurance"
                #         self.insurance_coverage = 0
                #         self.elevation = elevation
                #         self.EAD = EAD

                for type in ["NFIP", "private"]:
                    building_rate, content_rate = functions.insurance_rate(
                        type,
                        self.property_flood_zone,
                        self.new_property_height - self.BFE,
                        self.model.CRS_rewards,
                    )

                    if type == "NFIP":
                        insurance_coverage_options = self.model.NFIP_coverage_options.copy()
                    if type == "private":
                        insurance_coverage_options = self.model.private_coverage_options.copy()

                    if self.require_insurance == False:
                        insurance_coverage_options.append(0)
                        print('insurance_coverage_options', insurance_coverage_options)

                    for coverage in insurance_coverage_options:

                        insurance_cost = coverage * building_rate / 100
                        # print('insurance_cost', insurance_cost)
                        # print('building_rate', building_rate)
                        # print('coverage', coverage)

                        total_annual_cost = insurance_cost + annual_elevation_cost

                        PU, EAD, damage_list = functions.prospect_utility_action(
                            self.flood_elevation_list,
                            self.model.return_period_list,
                            self.new_property_height,
                            self.building_type,
                            self.risk_perception,
                            total_annual_cost,
                            self.house_value,
                            coverage,
                            self.public_risk_reduction,
                            False,
                        )

                        # print(type, PU, EAD, total_annual_cost)

                        if PU < self.PU:

                            self.PU = PU
                            self.elevation = elevation

                            self.insurance_coverage = coverage
                            self.EAD = EAD
                            self.damage_list = damage_list
                            if type == 'NFIP':
                                self.insurance_type = "NFIP"
                            if type == 'private':
                                self.insurance_type = "private"
