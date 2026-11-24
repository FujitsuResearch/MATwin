import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time
import isuelogit as isl
import tensorflow as tf
from datetime import datetime
from mate.metrics import mse, mape, r2_score,  z2score, mdape

from mate.visualizations import plot_flow_vs_traveltime, plot_relative_gap_by_period, \
    plot_equilibrium_loss_by_period, plot_utility_parameters_periods, plot_rr_by_period,\
        plot_parameters, plot_top_od_flows_periods, plot_performance_functions,\
            plot_metrics_kfold, plot_parameters_kfold, plot_baselines_kfold, \
                plot_predictive_performance, plot_convergence_estimates, plot_rr_by_period_models, \
                plot_total_trips_models

from mate.models import compute_baseline_predictions_kfold, compute_benchmark_metrics

#postprocessing_metrics
class postprocessing_metrics:
    def __init__(self, model, obs_flow=None, obs_tt=None):
        """
        Initialize the class with required data to calcualte different metrics.
        """
        self.model=model

        self.obs_flow=obs_flow
        self.obs_tt=obs_tt

        self.estimated_flow=None
        self.estimated_tt=None
    
    def compute_predictions(self, x_data, y_data, node_data, loss_metric, pretrain_link_flows,
                            optimizer, threshold_relative_gap, loss_weights, epochs, eval_metric=None,
                            training_flag=False, xt_val=None, yt_val=None):

        _ = self.model.predict(x_data,
                               node_data=node_data,
                               loss_metric=loss_metric,
                               pretrain_link_flows = pretrain_link_flows,
                               optimizer= optimizer,
                               loss_weights=loss_weights,
                               threshold_relative_gap= threshold_relative_gap,
                               epochs=epochs)

        if training_flag:
            # For Training file
            metrics_df = self.model.compute_loss_metrics(
                metrics={eval_metric.__name__: eval_metric, 'mse': mse, 'mape': mape, 'r2': r2_score},
                X=xt_val, Y=yt_val).assign(dataset='validation')
            metrics_df = pd.concat([metrics_df,
                                self.model.compute_loss_metrics(
                                    metrics={eval_metric.__name__: eval_metric, 'mse': mse, 'mape': mape, 'r2': r2_score},
                                    X=x_data, Y=y_data).assign(dataset='training'),
                                compute_benchmark_metrics(
                                    metrics={eval_metric.__name__: eval_metric, 'mse': mse, 'mape': mape, 'r2': r2_score},
                                    Y_ref=y_data, Y=yt_val).assign(dataset='benchmark')
                                ])
            with pd.option_context('display.float_format', '{:0.3g}'.format):
                print(pd.pivot(metrics_df, index=['component', 'dataset'], 
                           columns=['metric'])['value'])

        else:
            # For validation
            with pd.option_context('display.float_format', '{:0.3g}'.format):
                print('\n')
                metrics_df = self.model.compute_loss_metrics(
                        metrics={eval_metric.__name__: eval_metric, 'mse': mse, 'mape': mape, 'r2': r2_score},
                        X=x_data, Y=y_data)
                print(metrics_df)
        return metrics_df  

        # This can be skipped as the plotting functions internally calling these functions for predictions
        self.estimated_flow=self.model.predict_flow()
        self.estimated_tt=self.model.predict_traveltime()
    
    # Plotting traffic flow vs travel time
    def plot_flow_vs_traveltime(self, period_col, save_path, area_of_interest, 
                                plot_name, hour_label=True, all_metrics=False, plot_speed=False):
        
        fig, axs = plot_flow_vs_traveltime(model = self.model,
                        observed_traveltime=self.obs_tt,
                        observed_flow= self.obs_flow,
                        period_col = period_col,
                        hour_label=hour_label,
                        all_metrics = all_metrics,
                        plot_speed=False
                        )
        #Check the folder exists 
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        plt.savefig(save_path+f'{area_of_interest}-{plot_name}.png', dpi = 300, bbox_inches='tight')
        plt.show()

    def plot_all_predictive_performance(self, train_results_df, val_results_df, eval_metric, 
                                        save_path, area_of_interest):

        plot_predictive_performance(train_losses=train_results_df,
                                    val_losses=val_results_df,
                                    curves=["travel time", "link flow", "equilibrium"],
                                    xticks_spacing=2,
                                    show_validation=False)
        
        plt.savefig(save_path+f"{area_of_interest}-relative-mse-mate.png", dpi=300)

        plot_predictive_performance(train_losses=train_results_df,
                                    val_losses=val_results_df,
                                    curves=["travel time", "link flow", "equilibrium"],
                                    xticks_spacing=2,
                                    show_validation=False,
                                    show_percentage_units=True,
                                    prefix_metric=eval_metric.__name__,
                                    yaxis_label=f"{eval_metric.__name__}",)
        
        plot_predictive_performance(train_losses=train_results_df,
                                    val_losses=val_results_df,
                                    curves=["travel time", "link flow", "equilibrium"],
                                    xticks_spacing=2,
                                    show_validation=True,)

        plot_predictive_performance(train_losses=train_results_df,
                                    val_losses=val_results_df,
                                    xticks_spacing=2,
                                    show_validation=True,
                                    show_percentage_units=True,
                                    curves=["travel time", "link flow", "equilibrium"],
                                    prefix_metric=eval_metric.__name__,
                                    yaxis_label=f"{eval_metric.__name__}",)

    def plot_convergence_estimates(self, train_results_df_mate, save_path, area_of_interest):

        fig, ax = plot_convergence_estimates(
            estimates=train_results_df_mate.assign(relative_gap=np.abs(train_results_df_mate["relative_gap"]))[
                ["epoch", "relative_gap"]
                ],
                xticks_spacing=2,)

        ax.set_yscale("log")
        ax.set_ylabel("relative gap (log scale)")
        ax.get_legend().remove()
        plt.title("Relative gap", fontsize=20)
        plt.savefig(save_path+f"{area_of_interest}-relative-gap-mate.png", dpi=300)
        plt.show()

    def plot_relative_gap_by_period(self, period_keys, save_path, area_of_interest):
        plot_relative_gap_by_period(self.model, period_keys)
        plt.title("relative-gap-period", fontsize=20)
        plt.savefig(save_path+f"{area_of_interest}-relative-gap-period-mate.png", dpi=300)
        plt.show()

    def plot_equilibrium_loss_by_period(self, period_keys,save_path, area_of_interest):
        plot_equilibrium_loss_by_period(self.model, period_keys=period_keys)
        plt.title("equilibrium-loss-period", fontsize=20)
        plt.savefig(save_path+f"{area_of_interest}-equilibrium-loss-period-mate.png", dpi=300)
        plt.show()

    def plot_rr_by_period(self, period_keys, save_path, area_of_interest):
        # Compute utility parameters over time (heatmap) and value of travel time reliability (lineplot)
        theta_df = plot_utility_parameters_periods(self.model, period_keys=period_keys, 
                                                   period_feature="hour")
        if "tt_sd" in theta_df.keys():
            plot_rr_by_period(self.model,
                              period_keys,
                              model_key="MaTE",
                              period_feature="hour",
                              rotation_xticks=90,
                              )
            plt.legend().remove()
            plt.ylim(bottom=-0.1)
            plt.title("reliability-ratios-periods", fontsize=20)
            plt.savefig(save_path+f"{area_of_interest}-reliability-ratios-periods-mate.png", dpi=300)
            plt.show()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
        sns.histplot(pd.DataFrame({"fixed_effect": np.array(self.model.fixed_effect)}),
                     stat="density",
                     x="fixed_effect",
                     alpha=0.8,
                     ax=ax,
                     binwidth=0.5
                     )
        plt.show()

    def plot_parameters_by_period(self, period_keys, save_path, area_of_interest):

        # Plot parameters by period
        theta_df = (
            plot_utility_parameters_periods(self.model, 
                                            period_keys=period_keys, 
                                            period_feature="hour", 
                                            plot=False).reset_index().
                                            drop(["hour", "index"], axis=1).
                                            reset_index().rename(columns={"index": "period"}))

        theta_df = pd.melt(theta_df, id_vars="period", var_name="parameter").assign(group="utility")

        kappa_df = pd.DataFrame({})
        if self.model.kappa is not None:
            kappa_df = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "period": [period],
                            "parameter": [feature],
                            "value": [self.model.kappa[period, i].numpy()],
                        }
                    )
                    for period in range(self.model.kappa.shape[0])
                    for i, feature in enumerate(self.model.generation.features)
                ]).assign(group="generation")

        parameters_df = pd.concat([theta_df, kappa_df]).replace(
            {
                "tt": "travel time",
                "tt_sd": "std. travel time",
                "s": "intersections",
                "bus_stops": "bus stops",
                "median_inc": "income",
                "rr": "reliability ratio",
            })

        parameters_df["hour"] = parameters_df["period"].map(
            {v: period_keys[period_keys.period_id == k]["hour"].iloc[0] for k, v in self.model.period_dict.items()}
            )

        fig, ax = plot_parameters(
            df=parameters_df[parameters_df.group == "utility"],
            n_cols_legend=3,
            figsize=(5.5, 6),
            hour_label=True,
            rotation_xticks=90,)

        plt.axhline(0, linestyle="dashed", color="black")
        plt.title("utility-periods", fontsize=20)
        plt.savefig(save_path+f"{area_of_interest}-utility-periods-mate.png", dpi=300)
        plt.show()

        print("Theta:", self.model.theta.numpy().T)
        if self.model.kappa is not None:
            print("Kappa:", self.model.kappa.numpy().T)
    
    def plot_top_od_flows_periods(self, period_keys, reference_q, save_path, area_of_interest):

        # Plot heatmap with flows of top od pairs
        plot_top_od_flows_periods(self.model,
                                  reference_od=reference_q[0],
                                  period_keys=period_keys,
                                  period_feature="hour",
                                  top_k=20,
                                  join_points=True,
                                  rotation_xticks=90,)
        
        plt.title("total-trips-periods", fontsize=20)
        plt.savefig(save_path+f"{area_of_interest}-total-trips-periods-mate.png", dpi=300)
        plt.show()
    
    def plot_performance_function(self, network, save_path, area_of_interest, alpha=0.15, 
                                  beta=4, marginal=False, flow_range=range(0,12000,100), 
                                  sharey=False):

        plot_performance_functions(model=self.model,
                                   network=network,
                                   marginal=marginal,
                                   alpha=alpha,
                                   beta=beta,
                                   sharey=sharey,
                                   flow_range=flow_range,)
        plt.savefig(save_path+f"{area_of_interest}-comparison-all-link-performance-functions-mate.png",
        dpi=300)
        plt.show()

        # To generate the simulation.txt file.
    def perform_simulation(self, save_path, bef_flag, itr):
        
        flow = self.model.predict_flow() #veh/hr
        tt = self.model.predict_traveltime() #seconds

        file_content = f"""************ travel stats ************
car count: {np.sum(np.sum(self.model.q, axis=1))}
truck count: {-1}
car total travel time (hours): {np.sum(tf.multiply(tt/3600,flow))}
truck total travel time (hours): {-1}
average vehicle delay (minutes): {cal_avg_vehicle_delay(self.model)}
average car delay (minutes): {cal_avg_vehicle_delay(self.model)}
average truck delay (minutes): {-1}
************ travel stats ************
            """

        # Write the content to a .txt file
        sim_file_name = "simulation_baseline.txt"
        if bef_flag==False:
            iteration = itr
            sim_file_name = f"simulation_intervention_{iteration}.txt"

        with open(f'{save_path}/{sim_file_name}', "w") as file:
            file.write(f"{file_content}\n")
        print(f"File '{sim_file_name}' has been created.")

    # Function to calculate fuel & pollutants emission
    def perform_emission(self, save_path, bef_flag, itr):

        hours=self.model.period_ids.shape[0]

        flow = self.model.predict_flow() #veh/hr
        tt = self.model.predict_traveltime() #seconds

        # length unit change from meters to miles
        links_len=np.tile(self.model.network.links_len*0.0006, (hours, 1))

        vmt=np.sum(links_len*flow)

        vehicle_types=["cars", "trucks"]

        # Fuel efficiency in miles per gallon
        fuel_efficiency_cars = 2759578.127027/100838.586188 #27
        fuel_efficiency_trucks = 692439.911966/39534.538121 #17

        # Emission factors in grams per gallon
        #https://nepis.epa.gov/Exe/tiff2png.cgi/P100JPPH.PNG?-r+75+-g+7+D%3A%5CZYFILES%5CINDEX%20DATA%5C11THRU15%5CTIFF%5C00000730%5CP100JPPH.TIF
        #https://epa.illinois.gov/topics/air-quality/planning-reporting/annual-emission-reports/reference-tables/common-emission-factors.html
        # According to the U.S. Environmental Protection Agency (EPA), burning one gallon of gasoline produces approximately 8,887 grams of CO₂.
        # The value of 0.1 grams HC per gallon is a generalized estimate.
        # The value of 5.0 grams CO per gallon is an approximate average.
        # 0.08 g NOₓ per mile for modern passenger vehicles. 0.08*25miles/gallon
        """emission_factors_cars = {
            "CO2": 8887,  # CO2 in grams per gallon
            "HC": 0.1,    # Hydrocarbons in grams per gallon
            "CO": 5.0,    # Carbon Monoxide in grams per gallon
            "NOX": 2.0    # Nitrogen Oxides in grams per gallon
        }"""

        # current values are obtained from the emission.txt file from MAC-POSTS
        emission_factors_cars = {
            "CO2": 896152515.462331/100838.586188, #8887,  # CO2 in grams per gallon
            "HC": 1983095.690643/100838.586188, #19.66   # Hydrocarbons in grams per gallon
            "CO": 3622417.854603/100838.586188, #35.92    # Carbon Monoxide in grams per gallon
            "NOX": 3149309.920131/100838.586188 #31.23    # Nitrogen Oxides in grams per gallon
        }

        emission_factors_trucks = {
            "CO2": 351343440.275302/39534.538121, # CO2 in grams per gallon
            "HC": 937868.986916/39534.538121, # Hydrocarbons in grams per gallon
            "CO": 4395194.459914/39534.538121, # Carbon Monoxide in grams per gallon
            "NOX": 4695661.954355/39534.538121 # Nitrogen Oxides in grams per gallon
        }

        # Calculate fuel gallons consumed
        fuel_gallons_cars = vmt / fuel_efficiency_cars
        #fuel_gallons_trucks= vmt / fuel_efficiency_trucks

        # Calculate pollutants emissions
        emissions_cars = {pollutant: fuel_gallons_cars * grams for pollutant, grams in 
                        emission_factors_cars.items()}

        #emissions_trucks={pollutant: fuel_gallons_trucks * grams for pollutant, grams in emission_factors.items()}

        for vehicle in vehicle_types:
            if vehicle=="cars":
                file_content_car = f"""The emission stats for {vehicle} are:
fuel: {np.sum(vmt)/fuel_efficiency_cars} gallons, CO2: {emissions_cars['CO2']} g, HC: {emissions_cars['HC']} g, CO: {emissions_cars['CO']} g, NOX: {emissions_cars['NOX']} g, \
Total Car VMT: {np.sum(vmt)} miles, EV Car VMT: {-1} miles, VHT: {np.sum(tf.multiply(tt/3600,flow))} hours, number of trips: {np.sum(np.sum(self.model.q, axis=1))} trips
"""

            if vehicle=="trucks":
                # Before intervention
                file_content_truck = f"""The emission stats for {vehicle} are:
fuel: {-1} gallons, CO2: {-1} g, HC: {-1} g, CO: {-1} g, NOX: {-1} g, 
Total Car VMT: {-1} miles, EV Car VMT: {-1} miles, VHT: {-1} hours, number of trips: {-1} trips
"""

         # Write the content to a .txt file
        em_file_name = "emission_baseline.txt"
        if bef_flag==False:
            iteration = itr
            em_file_name = f"emission_intervention_{iteration}.txt"

        with open(f'{save_path}/{em_file_name}', "w") as file:
            file.write(f"{file_content_car}")
            file.write(f"{file_content_truck}")
        print(f"File '{em_file_name}' has been created.")


# Function to calculat average vehicle delay in minutes
def cal_avg_vehicle_delay(model):

    hours=model.period_ids.shape[0]

    #free flow travel time
    ff_tt=np.tile([link.Z_dict["tf"] for link in model.network.links], (hours, 1))
        
    flow = model.predict_flow() #veh/hr
    tt = model.predict_traveltime() #seconds

    bf_average_vehicle_delay=np.sum(np.sum(tf.multiply((tt-ff_tt)/60,flow), axis=1))/np.sum(np.sum(model.q, axis=1))

    return bf_average_vehicle_delay

# All model functions
"""
# Plot of convergence toward true rr across models
def plot_models_convergence_plot(models, theta_df, train_results_dfs, save_path, area_of_interest):

    # Plot of convergence toward true rr across models
    train_estimates = {}
    train_losses = {}

    for model_key, model in models.items():
        train_estimates[model_key], train_losses[model_key] = model.split_results(results=train_results_dfs[model_key])
        train_estimates[model_key]["model"] = model_key

    train_estimates_df = pd.concat(train_estimates.values())
    train_estimates_df["rr"] = train_estimates_df["tt"]
    if "tt_sd" in theta_df.keys():
        train_estimates_df["rr"] = train_estimates_df["tt_sd"] / train_estimates_df["tt"]

    estimates = train_estimates_df[["epoch", "model", "rr"]].reset_index().drop("index", axis=1)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    g = sns.lineplot(data=estimates, x="epoch", hue="model", y="rr")
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)
    ax.set_ylabel("reliability ratio", fontsize=18)
    ax.set_xticks(np.arange(estimates["epoch"].min(), estimates["epoch"].max() + 1, 2))
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.legend(fontsize=14, loc="upper right", title_fontsize=14, title="model")
    plt.show()

# Plot of reliability ratio by hour for all models
def plot_models_rr_by_period(models, theta_df, period_keys, save_path, area_of_interest):
    if "tt_sd" in theta_df.keys():
        reliability_ratios = plot_rr_by_period_models(models, period_keys, period_feature="hour", join=True, rotation_xticks=90)
            
        plt.savefig(save_path+f"{area_of_interest}-reliability-ratios-periods.png", dpi=300)
        plt.show()
        print(reliability_ratios)

    if "tt_sd" in theta_df.keys():
        with pd.option_context("display.float_format", "{:0.4f}".format):
            print(reliability_ratios[reliability_ratios.model == "MaTE"][["hour", "rr"]].T)

    if "tt_sd" in theta_df.keys():
        print(reliability_ratios.groupby("model")[["rr"]].mean().round(4))
"""
#Plot of total trips by hour for all models
def plot_models_total_trips(models, network, period_keys, save_path, area_of_interest):

    total_trips = plot_total_trips_models(models=models,
                                          period_feature="hour",
                                          period_keys=period_keys,
                                          reference_od=network.q.flatten(),
                                          join=True,
                                          rotation_xticks=90,)
    plt.title("total-trips-periods", fontsize=20)
    plt.savefig(save_path+f"{area_of_interest}-total-trips-periods.png", dpi=300)
    plt.show()
    print(total_trips)
    print(total_trips.groupby("model")[["total_trips"]].sum().round(1))

# Validation functions
def plot_metrics_kfold_val(kfold_model, save_path, area_of_interest, eval_metric):
        
    fig, axs = plot_metrics_kfold(df = kfold_model[kfold_model.component.isin(['flow','traveltime'])], model_name = 'MaTE', metric_name = eval_metric.__name__, showfliers = True)
    
    for ax in axs.reshape(-1):
        ax.set_ylim(bottom = 0)
    plt.title("kfold metrics", fontsize=20)
    plt.savefig(save_path+f'{area_of_interest}-kfold-{eval_metric.__name__}-mate.png', dpi = 300)
    plt.show()


def plot_parameters_kfold_val(kfold_model, parameters_kfold_df, period_keys, save_path, area_of_interest):

    parameters_kfold_df['MaTE']['hour'] = parameters_kfold_df['MaTE'].period.replace({v: k for k, v in kfold_model.period_dict.items()}). \
    replace(dict(zip(period_keys.period_id, period_keys.hour)))

    parameters_kfold_df['MaTE']['parameter'] = parameters_kfold_df['MaTE']['parameter'].\
    replace({'tt': 'travel time', 'median_inc': 'income', 'tt_sd': 'std. travel time', 'bus_stops': 'bus stops', 'vot': 'reliability ratio'})

    parameters_kfold_df['MaTE']['hour'] \
    = parameters_kfold_df['MaTE']['period'].map({v:period_keys[period_keys.period_id == k]['hour'].iloc[0] for k,v in kfold_model.period_dict.items()})

    parameters_kfold_df['MaTE'] = parameters_kfold_df['MaTE'][parameters_kfold_df['MaTE'].parameter != 'reliability ratio']

    fig, axs = plot_parameters_kfold(df = parameters_kfold_df['MaTE'][parameters_kfold_df['MaTE'].group == 'utility'], 
                                     n_cols_legend = 2, figsize = (5.5,5.5), 
                                     hour_label = True, rotation_xticks=90)
    plt.axhline(0, linestyle ='dashed', color = 'black')
    plt.title("kfold-utility-periods", fontsize=20)
    plt.savefig(save_path+f'{area_of_interest}-kfold-utility-periods-mate.png', dpi = 300)
    plt.show()

    fig, axs = plot_parameters_kfold(df = parameters_kfold_df['MaTE'][parameters_kfold_df['MaTE'].group == 'generation'],
                                    n_cols_legend = 3, figsize = (5.5,5.5), 
                                    hour_label = True, rotation_xticks=90)
    plt.axhline(0, linestyle ='dashed', color = 'black')
    plt.title("kfold-generation-periods", fontsize=20)
    plt.savefig(save_path+f'{area_of_interest}-kfold-generation-periods-mate.png', dpi = 300)
    plt.show()


def plot_baselines_kfold_val(links_gdf, metrics_kfold_df, XT_val, YT_val, eval_metric, 
                         n_splits, seed, network, save_path, area_of_interest):

    # Calculate centroids of links to compute baselines models
    links_gdf['X'] = links_gdf.to_crs(2228).geometry.centroid.to_crs(4326).x
    links_gdf['Y'] = links_gdf.to_crs(2228).geometry.centroid.to_crs(4326).y
    coordinates = links_gdf[['X', 'Y']].values

    
    # Baselines are computed for every sample
    metrics_kfold_df['baselines'] = pd.DataFrame({})
    n_samples = XT_val.shape[0]

    t0 = time.time()
    for i in range(n_samples):

        isl.printer.printIterationBar(i+1, n_samples, prefix='Sample:',length=20)

        metrics_kfold_df['baselines'] = pd.concat([metrics_kfold_df['baselines'],
                                    compute_baseline_predictions_kfold(
                                        metric=eval_metric,
                                        X = XT_val[i],
                                        y = YT_val[i][:,0][tf.newaxis,:],
                                        coordinates = coordinates,
                                        n_splits = n_splits, seed = seed).assign(component = 'traveltime')]) 
        metrics_kfold_df['baselines'] = pd.concat([metrics_kfold_df['baselines'],
                                    compute_baseline_predictions_kfold(
                                        metric=eval_metric,
                                        X = XT_val[i],
                                        y = YT_val[i][:,1][tf.newaxis,:],
                                        coordinates = coordinates,
                                        n_splits = n_splits, seed = seed).assign(component = 'flow')])

        metrics_kfold_df['baselines']['sample'] = i

    print(f'runtime: {time.time()-t0:0.1f} [s]')

    #Check the folder exists 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    ts = datetime.now().strftime('%y%m%d%H%M%S')
    metrics_kfold_df['baselines'].to_csv(save_path+f"{ts}_kfold_baselines_{network.key}.csv")

    # Comparison between our model and benchmarks
    metrics_kfold_plot_df = metrics_kfold_df['MaTE'][
        (metrics_kfold_df['MaTE'].component != 'equilibrium') & (metrics_kfold_df['MaTE'].metric == eval_metric.__name__) &
        (metrics_kfold_df['MaTE'].dataset == 'validation') & (metrics_kfold_df['MaTE'].stage == 'final')].\
        assign(model = 'MaTE')

    #Check the folder exists 
    if not os.path.exists(save_path+"/output/tables"):
        os.makedirs(save_path+"/output/tables")

    metrics_kfold_df['baselines'] \
    = metrics_kfold_df['baselines'].groupby(['fold', 'model', 'component'])[['value']].mean().reset_index()

    metrics_kfold_plot_df = pd.concat([metrics_kfold_df['baselines'][['model','component','value']],
                                    metrics_kfold_plot_df[['model','component','value']]])

    metrics_kfold_plot_df.loc[metrics_kfold_plot_df.model == 'MaTE', 'model'] = 'our model'

    #metrics_kfold_plot_df.to_csv(f"./output/tables/{ts}_kfold_{network.key}_all.csv")
    metrics_kfold_plot_df.to_csv(save_path+f"/output/tables/{ts}_kfold_{network.key}_all.csv")

    means_mapes = metrics_kfold_plot_df.groupby(['component','model'])['value'].mean().reset_index().rename(columns={'value':'mean'})
    se_mapes = metrics_kfold_plot_df.groupby(['component','model'])['value'].agg(['std', 'count']).apply(lambda row: row['std'] / np.sqrt(row['count']), axis=1).reset_index().rename(columns={0:'se'})

    pd.merge(means_mapes, se_mapes).sort_values(by = ['component', 'mean'])

    # Note: Result from ordinary kriging are removed due to unstable estimates and extremely large values of evaluation metric for travel time
    fig, _ = plot_baselines_kfold(df = metrics_kfold_plot_df.
                        assign(model = metrics_kfold_plot_df.model.replace('_', ' ', regex=True)).replace('our model', 'MaTE', regex=True)\
                            [metrics_kfold_plot_df.model != 'ordinary_kriging'],
                        metric_name=eval_metric.__name__, sharex=True, sharey=True, showfliers = False)

    fig.savefig(save_path+f"{area_of_interest}-kfold-{eval_metric.__name__}-baselines-mate.png", dpi=300)


                        

    