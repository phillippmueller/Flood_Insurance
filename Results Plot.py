# %%
import geopandas as gpd
from shapely import wkt
import pandas as pd
import matplotlib.pyplot as plt
import mapclassify


# %%

census_tract_number_list = [0, 10, 25, 50]
policy_list = ["pre_FIRM", "voucher"]


def plot_subplots(policy_name, census_tract_number):
    result_df = pd.read_csv(
        "data/result_{}_{}.csv".format(policy_name, census_tract_number)
    )
    result_df["geometry"] = result_df["geometry"].apply(wkt.loads)
    structures_gdf = gpd.GeoDataFrame(result_df, geometry="geometry", crs="EPSG:4326")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Simulation_Result_{}_{}".format(policy_name, census_tract_number))

    column_title_map = {
        "EAD": "EAD",
        "insurance_type": "Insurance Type",
        "insurance_coverage": "Insurance Coverage",
        "elevation": "Elevation",
    }
    maker_size = 1
    for ax, (column, title) in zip(axes.flatten(), column_title_map.items()):
        if column == "EAD":
            structures_gdf.plot(
                column=column,
                legend=True,
                aspect=1,
                cmap="viridis_r",
                markersize=maker_size,
                ax=ax,
                scheme="quantiles",
            )

        elif column in ["insurance_coverage", "elevation"]:
            plot = structures_gdf.plot(
                column=column,
                legend=False,
                aspect=1,
                cmap="viridis_r",
                markersize=maker_size,
                ax=ax,
            )
            cbar = plot.get_figure().colorbar(
                plot.collections[0], ax=ax, shrink=0.5, pad=-0.13
            )
            cbar.ax.tick_params(labelsize=8)
            # fig.subplots_adjust(right=1)

        else:
            structures_gdf.plot(
                column=column,
                legend=True,
                aspect=1,
                cmap="viridis_r",
                markersize=maker_size,
                ax=ax,
            )

        ax.set_title(title)

    plt.savefig(
        "plots/{}_{}.png".format(policy_name, census_tract_number),
        dpi=300,
        bbox_inches="tight",
    )


plot_subplots("voucher", 0)

# %%

for census_tract_number in census_tract_number_list:
    for policy_name in policy_list:
        plot_subplots(policy_name, census_tract_number)

# %%
for census_tract_number in census_tract_number_list:
    for policy_name in policy_list:
        result_df = pd.read_csv(
            "data/result_{}_{}.csv".format(policy_name, census_tract_number)
        )
        if census_tract_number_list.index(census_tract_number) > 0:
            previous_census_tract_number = census_tract_number_list[
                census_tract_number_list.index(census_tract_number) - 1
            ]

            previous_result_df = pd.read_csv(
                "data/result_{}.csv".format(previous_census_tract_number)
            )
            previous_result_df["geometry"] = previous_result_df["geometry"].apply(
                wkt.loads
            )
            previous_structures_gdf = gpd.GeoDataFrame(
                previous_result_df, geometry="geometry", crs="EPSG:4326"
            )

            result_df["geometry"] = result_df["geometry"].apply(wkt.loads)
            structures_gdf = gpd.GeoDataFrame(
                result_df, geometry="geometry", crs="EPSG:4326"
            )

            # Merge current and previous GeoDataFrames on geometry
            merged_gdf = structures_gdf.merge(
                previous_structures_gdf[["geometry", "EAD"]],
                on="geometry",
                suffixes=("", "_previous"),
            )

            # Calculate the difference in EAD
            merged_gdf["EAD_difference"] = (
                merged_gdf["EAD_previous"] - merged_gdf["EAD"]
            )

            # Plot the difference
            ax = merged_gdf.plot(
                column="EAD_difference",
                legend=True,
                aspect=1,
                cmap="viridis_r",
                markersize=5,
            )

            # Add title to the figure
            plt.title(
                "EAD Difference Census Tract: {} vs {}".format(
                    census_tract_number, previous_census_tract_number
                )
            )

            plt.show()
    # result_df = result_df.head(100)
    # result_df = result_df[result_df["EAD"] < 10000]

    # Determine the groups using quantiles

    # Plot the locations on the map with color based on the determined groups

    # Add title to the figure
