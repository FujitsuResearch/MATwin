import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import os 

#post intervention metrics
class postinterventions_metrics:
    def __init__(self, model, links_df, bf_flow, af_flow, bf_tt, af_tt, hours, 
                 reports_path, area_of_interest, target_date, intrv_type=None, closure_links_indices=None, 
                 toll_links_indices=None, toll_prices=None, enhanced_links_indices=None, new_links_df=None, iteration=0):
        """
        Initialize the class with required data to calcualte different metrics.
        """
        self.model=model

        self.links_df= links_df
        # reshaping the links length data
        len_reshaped = links_df["length"].values.reshape(1, len(links_df["length"]))

        # To convert into m/sec to miles/hr
        self.bf_speed=(len_reshaped/bf_tt)*2.24
        self.af_speed=(len_reshaped/af_tt)*2.24  

        # estimated flow & travel time
        self.bf_flow=bf_flow
        self.af_flow=af_flow
        self.bf_tt=bf_tt
        self.af_tt=af_tt

        # Length from meters to miles
        self.bf_miles=(len_reshaped*.0006)*bf_flow
        self.af_miles=(len_reshaped*.0006)*af_flow

        # hours
        self.hours=hours
        # reports save path
        self.reports_path=reports_path
        # area of interest
        self.area_of_interest=area_of_interest
        # Target date
        self.target_date=target_date
        # type of intervention
        self.intrv_type=intrv_type
        # closure links indices
        self.closure_links_indices=closure_links_indices
        # toll links indices
        self.toll_links_indices=toll_links_indices
        # toll prices
        self.toll_prices=toll_prices
        # Capacity enhanced links indices
        self.enhanced_links_indices=enhanced_links_indices
        # new links df(mainly to fetch new_capacity)
        self.new_links_df=new_links_df
        
        #iteration of papermill
        self.iteration = iteration
        
    def plot_avg_speed(self):
        """Funtion to plot average speed in the network before intervention & after intervention
        """

        bf_speed_norm=tf.multiply(self.bf_speed,self.bf_flow)
        af_speed_norm=tf.multiply(self.af_speed,self.af_flow)

        speed_bar_df = pd.DataFrame({
                                    'Hour':  self.hours * int(bf_speed_norm.shape[0]/len(self.hours)),  # 72 timestamps, can be mapped to actual hours if needed
                                    f'Before {self.intrv_type}': np.sum(bf_speed_norm, axis=1)/np.sum(self.bf_flow, axis=1),
                                    f'After {self.intrv_type}': np.sum(af_speed_norm, axis=1)/np.sum(self.af_flow, axis=1),
                                    })
        speed_bar_df['Hour'] = speed_bar_df['Hour'].astype(str)
        speed_bar_df.to_csv(os.path.join(self.reports_path,f'avg_speed_{self.iteration}.csv'))
        
        plot_two_bars(speed_bar_df,
                      barlabel1=speed_bar_df.columns[1], 
                      barlabel2=speed_bar_df.columns[2], 
                      xlabel="Hour",
                      ylabel="Travel speed (miles/hr)",
                      title="System level average travel speed",
                      roundflg=False,
                      date=self.target_date,
                      folder_path=self.reports_path,
                      file_name=self.area_of_interest)

    def plot_avg_vmt(self):
        """Funtion to plot average vehicle miles travel in the network before intervention & after intervention
        """

        miles_bar_df = pd.DataFrame({
                                    'Hour':  self.hours * int(self.bf_miles.shape[0]/len(self.hours)), 
                                    f'Before {self.intrv_type}': np.sum(self.bf_miles, axis=1)/np.tile(np.sum(self.model.q, axis=1), 
                                                                                                      int(self.bf_miles.shape[0]/len(self.hours))), # divide by total trips (sum od matrix for that hour)
                                    f'After {self.intrv_type}': np.sum(self.af_miles, axis=1)/np.tile(np.sum(self.model.q, axis=1), 
                                                                                                      int(self.af_miles.shape[0]/len(self.hours)))
                                })
        miles_bar_df['Hour'] = miles_bar_df['Hour'].astype(str)
        miles_bar_df.to_csv(os.path.join(self.reports_path,f'avg_vmt_{self.iteration}.csv'))
        
        plot_two_bars(miles_bar_df,
                      barlabel1=miles_bar_df.columns[1], 
                      barlabel2=miles_bar_df.columns[2], 
                      xlabel="Hour",
                      ylabel="Vehicle miles travelled",
                      title="System level average vehicle miles travel by hour",
                      roundflg=False,
                      date=self.target_date,
                      folder_path=self.reports_path,
                      file_name=self.area_of_interest)

    def plot_total_vmt(self):
        """Funtion to plot total vehicle miles travel in the network before intervention & after intervention
        """

        miles_bar_df = pd.DataFrame({
                                'Hour':  self.hours * int(self.bf_miles.shape[0]/len(self.hours)),
                                f'Before {self.intrv_type}': np.sum(self.bf_miles, axis=1)/1000,
                                f'After {self.intrv_type}': np.sum(self.af_miles, axis=1)/1000,
                                })
        miles_bar_df['Hour'] = miles_bar_df['Hour'].astype(str)
        miles_bar_df.to_csv(os.path.join(self.reports_path,f'total_vmt_{self.iteration}.csv'))
        
        plot_two_bars(miles_bar_df, 
                      barlabel1=miles_bar_df.columns[1],
                      barlabel2=miles_bar_df.columns[2],
                      xlabel="Hour",
                      ylabel="Total vehicle miles travelled (thousands of miles)",
                      title="System level total vehicle miles travel by hour",
                      roundflg=False,
                      date=self.target_date,
                      folder_path=self.reports_path,
                      file_name=self.area_of_interest)

    def plot_vmt_diff(self):
        """Funtion to plot vehicle miles travel difference in the network after adding intervention
        """

        bf_vmt_avg= np.sum(self.bf_miles, axis=1)/np.tile(np.sum(self.model.q, axis=1), 
                                                          int(self.bf_miles.shape[0]/len(self.hours)))
        af_vmt_avg= np.sum(self.af_miles, axis=1)/np.tile(np.sum(self.model.q, axis=1), 
                                                          int(self.af_miles.shape[0]/len(self.hours)))

        vmt_change_bar_df = pd.DataFrame({
            'Hour':  self.hours * int(self.bf_miles.shape[0]/len(self.hours)),
            'vmt change': af_vmt_avg - bf_vmt_avg
        })
        vmt_change_bar_df['Hour'] = vmt_change_bar_df['Hour'].astype(str)
        vmt_change_bar_df.to_csv(os.path.join(self.reports_path,f'vmt_diff_{self.iteration}.csv'))
        
        plot_one_bar(vmt_change_bar_df,
                     barlabel1=vmt_change_bar_df.columns[1],
                     xlabel="Hour",
                     ylabel=f"Difference in vehicles miles travelled by traveler after {self.intrv_type}",
                     title="System level vehicle miles travel change by hour",
                     date=self.target_date,
                     folder_path=self.reports_path,
                     file_name=self.area_of_interest)
        
    def plot_closure_link_tf(self):
        """Funtion to plot traffic flow in the road closure link before intervention & after intervention
        """

        for i in range(0, len(self.closure_links_indices)):
            bf_flow_link=self.bf_flow[:,self.closure_links_indices[i]]
            af_flow_link=self.af_flow[:,self.closure_links_indices[i]]

            max_flow = self.links_df.iloc[self.closure_links_indices[i]]["capacity"]
            new_max_flow = self.new_links_df.iloc[self.closure_links_indices[i]]["capacity"]

            link_bar_df = pd.DataFrame({
                                        'Hour':  self.hours * int(bf_flow_link.shape[0]/len(self.hours)),
                                        f'Before {self.intrv_type}-{i+1}': bf_flow_link,
                                        f'After {self.intrv_type}-{i+1}': af_flow_link
                                        })

            link_bar_df['Hour'] = link_bar_df['Hour'].astype(str)
            link_bar_df.to_csv(os.path.join(self.reports_path,f'traffic_flow_{self.iteration}_{self.closure_links_indices[i]}.csv'))
            
            plot_two_bars(link_bar_df,
                          barlabel1=link_bar_df.columns[1], 
                          barlabel2=link_bar_df.columns[2],
                          xlabel="Hour",
                          ylabel="Traffic flow (veh/hr)",
                          title=f"Closure link {i+1} traffic flow",
                          roundflg=True,
                          date=self.target_date,
                          folder_path=self.reports_path,
                          file_name=self.area_of_interest, 
                          max_value=max_flow,
                          new_max_value=None if self.intrv_type=="Full_RoadClosure" else new_max_flow)

    def plot_enhanced_vc_ratio(self, intervention_hours=None):

        for i in range(0, len(self.enhanced_links_indices)):
            bf_flow_link=self.bf_flow[:,self.enhanced_links_indices[i]]
            af_flow_link=self.af_flow[:,self.enhanced_links_indices[i]]

            bf_cap = self.links_df.iloc[self.enhanced_links_indices[i]]["capacity"]
            
            # In case of time dependent intervention hours.
            if intervention_hours:
                link_bar_df = pd.DataFrame({
                                        'Hour':  self.hours * int(bf_flow_link.shape[0]/len(self.hours)),
                                        f'Before {self.intrv_type}-{i+1}': bf_flow_link/bf_cap,
                                        f'After {self.intrv_type}-{i+1}': af_flow_link/bf_cap
                                        })

                for hour in intervention_hours:
                    link_bar_df.loc[link_bar_df['Hour'] == hour, f'After {self.intrv_type}-{i+1}'] = af_flow_link.numpy()[hour]/self.new_links_df.iloc[self.enhanced_links_indices[i]]["capacity"]
            
            else:

                af_cap = self.new_links_df.iloc[self.enhanced_links_indices[i]]["capacity"]
                link_bar_df = pd.DataFrame({
                                        'Hour':  self.hours * int(bf_flow_link.shape[0]/len(self.hours)),
                                        f'Before {self.intrv_type}-{i+1}': bf_flow_link/bf_cap,
                                        f'After {self.intrv_type}-{i+1}': af_flow_link/af_cap
                                        })
                
            link_bar_df['Hour'] = link_bar_df['Hour'].astype(str)
            link_bar_df.to_csv(os.path.join(self.reports_path,f'enhanced_vc_ratio_{self.iteration}_{self.enhanced_links_indices[i]}.csv'))

            plot_two_bars(link_bar_df,
                          barlabel1=link_bar_df.columns[1], 
                          barlabel2=link_bar_df.columns[2],
                          xlabel="Hour",
                          ylabel="VC ratio",
                          title=f"Enhanced link {i+1} VC ratio",
                          roundflg=False,
                          date=self.target_date,
                          folder_path=self.reports_path,
                          file_name=self.area_of_interest,)
    
    def plot_closure_link_tt(self):
        """Funtion to plot travel time in the road closure link before intervention & after intervention
        """

        for i in range(0, len(self.closure_links_indices)):
            bf_tt_link=self.bf_tt[:,self.closure_links_indices[i]]
            af_tt_link=self.af_tt[:,self.closure_links_indices[i]]

            link_bar_df = pd.DataFrame({
                                        'Hour':  self.hours * int(bf_tt_link.shape[0]/len(self.hours)),
                                        f'Before {self.intrv_type}-{i+1}': bf_tt_link,
                                        f'After {self.intrv_type}-{i+1}': af_tt_link
                                        })

            link_bar_df['Hour'] = link_bar_df['Hour'].astype(str)
            link_bar_df.to_csv(os.path.join(self.reports_path,f'closure_link_tt_{self.iteration}_{self.closure_links_indices[i]}.csv'))
            
            plot_two_bars(link_bar_df,
                          barlabel1=link_bar_df.columns[1], 
                          barlabel2=link_bar_df.columns[2],
                          xlabel="Hour",
                          ylabel="Travel time (seconds)",
                          title=f"Closure link {i+1} travel time",
                          roundflg=False,
                          date=self.target_date,
                          folder_path=self.reports_path,
                          file_name=self.area_of_interest,)

    def plot_closure_link_speed(self):
        """Funtion to plot travel speed in the road closure link before intervention & after intervention
        """

        for i in range(0, len(self.closure_links_indices)):
            bf_speed_link=self.bf_speed[:,self.closure_links_indices[i]]
            af_speed_link=self.af_speed[:,self.closure_links_indices[i]]

            link_bar_df = pd.DataFrame({
                                    'Hour':  self.hours * int(af_speed_link.shape[0]/len(self.hours)),
                                    f'Before {self.intrv_type}-{i+1}': bf_speed_link,
                                    f'After {self.intrv_type}-{i+1}': af_speed_link
                                })
            link_bar_df['Hour'] = link_bar_df['Hour'].astype(str)
            link_bar_df.to_csv(os.path.join(self.reports_path,f'travel_speed_{self.iteration}_{self.closure_links_indices[i]}.csv'))
            
            plot_two_bars(link_bar_df,barlabel1=link_bar_df.columns[1], 
                          barlabel2=link_bar_df.columns[2],
                          xlabel="Hour",
                          ylabel="Speed (miles/hr)",
                          title=f"Closure link {i+1} speed by hour",
                          roundflg=False,
                          date=self.target_date,
                          folder_path=self.reports_path,
                          file_name=self.area_of_interest)

    def plot_total_tf(self):
        """Funtion to plot total traffic flow in the network before intervention & after intervention
        """

        bf_flow_hourly_sum = np.sum(self.bf_flow, axis=1)
        af_flow_hourly_sum = np.sum(self.af_flow, axis=1)

        flows_bar_df = pd.DataFrame({
                                    'Hour':  self.hours * int(bf_flow_hourly_sum.shape[0]/len(self.hours)),
                                    f'Before {self.intrv_type}': bf_flow_hourly_sum,
                                    f'After {self.intrv_type}': af_flow_hourly_sum,
                                })
        flows_bar_df['Hour'] = flows_bar_df['Hour'].astype(str)
        flows_bar_df.to_csv(os.path.join(self.reports_path,f'total_traffic_flow_{self.iteration}.csv'))

        plot_two_bars(flows_bar_df,
                      barlabel1=flows_bar_df.columns[1],
                      barlabel2=flows_bar_df.columns[2], 
                      xlabel="Hour",
                      ylabel="Total Traffic flow (veh/hr)",
                      title="System level total traffic flow",
                      roundflg=True,
                      date=self.target_date,
                      folder_path=self.reports_path,
                      file_name=self.area_of_interest)
        
    def plot_total_tt(self):
        """Funtion to plot total travel time in the network before intervention & after intervention
        """

        # Converted time to hours & flow is veh/hrs already
        bf_tt_total = tf.multiply(self.bf_tt/3600,self.bf_flow)
        af_tt_total = tf.multiply(self.af_tt/3600,self.af_flow)

        tt_total_bar_df = pd.DataFrame({
                                        'Hour':  self.hours * int(bf_tt_total.shape[0]/len(self.hours)),
                                        f'Before {self.intrv_type}': np.sum(bf_tt_total, axis=1),
                                        f'After {self.intrv_type}': np.sum(af_tt_total, axis=1)
                                    })
        
        tt_total_bar_df['Hour'] = tt_total_bar_df['Hour'].astype(str)
        tt_total_bar_df.to_csv(os.path.join(self.reports_path,f'total_travel_time_{self.iteration}.csv'))
        
        plot_two_bars(tt_total_bar_df,
                      barlabel1=tt_total_bar_df.columns[1], 
                      barlabel2=tt_total_bar_df.columns[2], 
                      xlabel="Hour",
                      ylabel="Total travel time (hours)",
                      title="System level total travel time",
                      roundflg=True,
                      date=self.target_date,
                      folder_path=self.reports_path,
                      file_name=self.area_of_interest)

    def plot_avg_tt(self):
        """Funtion to plot average travel time in the network before intervention & after intervention
        """

        # Converted time to hours & flow is veh/hrs already
        bf_tt_total=tf.multiply(self.bf_tt/60,self.bf_flow)
        af_tt_total=tf.multiply(self.af_tt/60,self.af_flow)

        tt_bar_df = pd.DataFrame({
                                'Hour':  self.hours * int(bf_tt_total.shape[0]/len(self.hours)),
                                f'Before {self.intrv_type}': np.sum(bf_tt_total, axis=1)/np.tile(np.sum(self.model.q, axis=1), 
                                                                                                 int(self.bf_miles.shape[0]/len(self.hours))),
                                f'After {self.intrv_type}': np.sum(af_tt_total, axis=1)/np.tile(np.sum(self.model.q, axis=1), 
                                                                                                 int(self.af_miles.shape[0]/len(self.hours)))
                            })
        tt_bar_df['Hour'] = tt_bar_df['Hour'].astype(str)
        tt_bar_df.to_csv(os.path.join(self.reports_path,f'avg_travel_time_{self.iteration}.csv'))
        
        plot_two_bars(tt_bar_df,
                      barlabel1=tt_bar_df.columns[1], 
                      barlabel2=tt_bar_df.columns[2], 
                      xlabel="Hour",
                      ylabel="Travel time (minutes)",
                      title="System level average travel time",
                      roundflg=False,
                      date=self.target_date,
                      folder_path=self.reports_path,
                      file_name=self.area_of_interest)

    def plot_toll_link_tf(self):
        """Funtion to plot traffic flow in the toll links before intervention & after intervention
        """

        for i in range(0, len(self.toll_links_indices)):
            bf_flow_link=self.bf_flow[:,self.toll_links_indices[i]]
            af_flow_link=self.af_flow[:,self.toll_links_indices[i]]

            link_bar_df = pd.DataFrame({
                                        'Hour':  self.hours * int(bf_flow_link.shape[0]/len(self.hours)),
                                        f'Before {self.intrv_type}-{i+1}': bf_flow_link,
                                        f'After {self.intrv_type}-{i+1}': af_flow_link
                                        })

            max_flow = self.links_df.iloc[self.toll_links_indices[i]]["capacity"]

            link_bar_df['Hour'] = link_bar_df['Hour'].astype(str)
            # print("<<<<<<<<<<<<<<<<<", self.iteration, self.toll_links_indices[i])
            link_bar_df.to_csv(os.path.join(self.reports_path,f'traffic_flow_in_toll_links_{self.iteration}_{self.toll_links_indices[i]}.csv'))
            
            plot_two_bars(link_bar_df,
                          barlabel1=link_bar_df.columns[1],
                          barlabel2=link_bar_df.columns[2],
                          xlabel="Hour",
                          ylabel="Traffic flow (veh/hr)",
                          title=f"Toll link {i+1} traffic flow",
                          roundflg=True,
                          date=self.target_date,
                          folder_path=self.reports_path,
                          file_name=self.area_of_interest,
                          max_value=max_flow)

    def plot_toll_link_tt(self):
        """Funtion to plot travel time in the road closure link before intervention & after intervention
        """

        for i in range(0, len(self.toll_links_indices)):
            bf_tt_link=self.bf_tt[:,self.toll_links_indices[i]]
            af_tt_link=self.af_tt[:,self.toll_links_indices[i]]

            link_bar_df = pd.DataFrame({
                                        'Hour':  self.hours * int(bf_tt_link.shape[0]/len(self.hours)),
                                        f'Before {self.intrv_type}-{i+1}': bf_tt_link,
                                        f'After {self.intrv_type}-{i+1}': af_tt_link
                                        })

            link_bar_df['Hour'] = link_bar_df['Hour'].astype(str)
            link_bar_df.to_csv(os.path.join(self.reports_path,f'travel_time_in_toll_links_{self.iteration}_{self.toll_links_indices[i]}.csv'))
            
            plot_two_bars(link_bar_df,
                          barlabel1=link_bar_df.columns[1], 
                          barlabel2=link_bar_df.columns[2],
                          xlabel="Hour",
                          ylabel="Travel time (seconds)",
                          title=f"Toll link {i+1} travel time",
                          roundflg=False,
                          date=self.target_date,
                          folder_path=self.reports_path,
                          file_name=self.area_of_interest,)

    def plot_toll_link_speed(self):
        """Funtion to plot travel speed in the toll links before intervention & after intervention
        """

        for i in range(0, len(self.toll_links_indices)):
            bf_speed_link=self.bf_speed[:,self.toll_links_indices[i]]
            af_speed_link=self.af_speed[:,self.toll_links_indices[i]]

            link_bar_df = pd.DataFrame({
                                    'Hour':  self.hours * int(af_speed_link.shape[0]/len(self.hours)),
                                    f'Before {self.intrv_type}-{i+1}': bf_speed_link,
                                    f'After {self.intrv_type}-{i+1}': af_speed_link
                                })
            link_bar_df['Hour'] = link_bar_df['Hour'].astype(str)
            link_bar_df.to_csv(os.path.join(self.reports_path,f'travel_speed_in_toll_links_{self.iteration}_{self.toll_links_indices[i]}.csv'))
            
            plot_two_bars(link_bar_df,barlabel1=link_bar_df.columns[1], 
                          barlabel2=link_bar_df.columns[2], 
                          xlabel="Hour",
                          ylabel="Speed (miles/hr)",
                          title=f"Toll link {i+1} speed by hour",
                          roundflg=False,
                          date=self.target_date,
                          folder_path=self.reports_path,
                          file_name=self.area_of_interest)

    def plot_toll_revenue(self, intervention_hours=None):
        """Funtion to plot toll revenues in the toll links.
        """

        for i in range(0, len(self.toll_links_indices)):
            toll_revenue=self.af_flow[:,self.toll_links_indices[i]] *self.toll_prices[i]

            revenue_bar_df = pd.DataFrame({
                    'Hour':  self.hours * int(toll_revenue.shape[0]/len(self.hours)),
                    f'Toll-{i+1} revenue': toll_revenue/1000,
                })

            # In case of time-dependent intervention, no toll revenue in other times.
            if intervention_hours:
                for hour in range(self.af_flow.shape[0]):
                    if hour not in intervention_hours:
                        revenue_bar_df.loc[hour, f'Toll-{i+1} revenue'] = 0

            revenue_bar_df['Hour'] = revenue_bar_df['Hour'].astype(str)
            revenue_bar_df[f'Toll-{i+1} revenue'] = round(revenue_bar_df[f'Toll-{i+1} revenue'],1)
            revenue_bar_df.to_csv(os.path.join(self.reports_path,f'toll_revenue_{self.iteration}_{self.toll_links_indices[i]}.csv'))
            
            plot_one_bar(revenue_bar_df,
                        barlabel1=revenue_bar_df.columns[1], 
                        xlabel="Hour",
                        ylabel="Revenue by hour (thousands of dollars)",
                        title=f"Toll {i+1} revenue",
                        date=self.target_date,
                        folder_path=self.reports_path,
                        file_name=self.area_of_interest)
            
    def plot_enhanced_link_tf(self):
        """Funtion to plot traffic flow in the enhanced link before intervention & after intervention
        """

        for i in range(0, len(self.enhanced_links_indices)):
            bf_flow_link=self.bf_flow[:,self.enhanced_links_indices[i]]
            af_flow_link=self.af_flow[:,self.enhanced_links_indices[i]]

            max_flow = self.links_df.iloc[self.enhanced_links_indices[i]]["capacity"]
            new_max_flow=self.new_links_df.iloc[self.enhanced_links_indices[i]]["capacity"]

            link_bar_df = pd.DataFrame({
                                        'Hour':  self.hours * int(bf_flow_link.shape[0]/len(self.hours)),
                                        f'Before {self.intrv_type}-{i+1}': bf_flow_link,
                                        f'After {self.intrv_type}-{i+1}': af_flow_link
                                        })

            link_bar_df['Hour'] = link_bar_df['Hour'].astype(str)
            link_bar_df.to_csv(os.path.join(self.reports_path,f'enhanced_traffic_flow_{self.iteration}_{self.enhanced_links_indices[i]}.csv'))
            
            plot_two_bars(link_bar_df,
                          barlabel1=link_bar_df.columns[1], 
                          barlabel2=link_bar_df.columns[2],
                          xlabel="Hour",
                          ylabel="Traffic flow (veh/hr)",
                          title=f"Capacity enhanced link {i+1} traffic flow",
                          roundflg=True,
                          date=self.target_date,
                          folder_path=self.reports_path,
                          file_name=self.area_of_interest, 
                          max_value=max_flow,
                          new_max_value=None if self.intrv_type=="Full_RoadClosure" else new_max_flow)

    def plot_enhanced_link_tt(self):
        """Funtion to plot travel time in the enhanced link before intervention & after intervention
        """

        for i in range(0, len(self.enhanced_links_indices)):
            bf_tt_link=self.bf_tt[:,self.enhanced_links_indices[i]]
            af_tt_link=self.af_tt[:,self.enhanced_links_indices[i]]

            link_bar_df = pd.DataFrame({
                                        'Hour':  self.hours * int(bf_tt_link.shape[0]/len(self.hours)),
                                        f'Before {self.intrv_type}-{i+1}': bf_tt_link,
                                        f'After {self.intrv_type}-{i+1}': af_tt_link
                                        })

            link_bar_df['Hour'] = link_bar_df['Hour'].astype(str)
            link_bar_df.to_csv(os.path.join(self.reports_path,f'enhanced_travel_time_{self.iteration}_{self.enhanced_links_indices[i]}.csv'))
            
            #For a particular day
            link_bar_df=link_bar_df.iloc[0:len(self.hours)]

            plot_two_bars(link_bar_df,
                          barlabel1=link_bar_df.columns[1], 
                          barlabel2=link_bar_df.columns[2],
                          xlabel="Hour",
                          ylabel="Travel time (seconds)",
                          title=f"Capacity enhanced link {i+1} travel time",
                          roundflg=False,
                          date=self.target_date,
                          folder_path=self.reports_path,
                          file_name=self.area_of_interest,)

    def plot_enhanced_link_speed(self):
        """Funtion to plot travel speed in the enhanced link before intervention & after intervention
        """

        for i in range(0, len(self.enhanced_links_indices)):
            bf_speed_link=self.bf_speed[:,self.enhanced_links_indices[i]]
            af_speed_link=self.af_speed[:,self.enhanced_links_indices[i]]

            link_bar_df = pd.DataFrame({
                                    'Hour':  self.hours * int(af_speed_link.shape[0]/len(self.hours)),
                                    f'Before {self.intrv_type}-{i+1}': bf_speed_link,
                                    f'After {self.intrv_type}-{i+1}': af_speed_link
                                })
            link_bar_df['Hour'] = link_bar_df['Hour'].astype(str)
            link_bar_df.to_csv(os.path.join(self.reports_path,f'enhanced_travel_speed_{self.iteration}_{self.enhanced_links_indices[i]}.csv'))
            
            plot_two_bars(link_bar_df,barlabel1=link_bar_df.columns[1], 
                          barlabel2=link_bar_df.columns[2],
                          xlabel="Hour",
                          ylabel="Speed (miles/hr)",
                          title=f"Capacity Enhanced link {i+1} speed by hour",
                          roundflg=False,
                          date=self.target_date,
                          folder_path=self.reports_path,
                          file_name=self.area_of_interest)


def plot_two_bars(plot_df, barlabel1, barlabel2, xlabel, ylabel, 
                      title, roundflg, date, folder_path, file_name, max_value=None, new_max_value=None):
    """
    Plot two bars for each time instance.
    """

    plt.figure(figsize=(16, 8))
    x = np.arange(len(plot_df[xlabel]))

    bar1= plt.bar(x-0.2, plot_df[barlabel1], width=0.4, label=barlabel1,)
    bar2= plt.bar(x+0.2, plot_df[barlabel2], width=0.4, label=barlabel2,)

    for bars in [bar1, bar2]:
        for bar in bars:
            yval = bar.get_height()
            if roundflg:
                yval= round(yval)
            plt.text(bar.get_x() + bar.get_width()/2, 
                    yval/2, 
                    round(yval,2), 
                    ha='center', 
                    va='bottom',
                    rotation=90,
                    fontsize=13)

    if max_value!=None:
        # Add a horizontal dotted line for the maximum flow
        plt.axhline(y=max_value, color='black', linestyle='--')
        # Add the label above the dotted line
        plt.text(0, (max_value+max_value*0.005), f'Link Capacity', 
                 color='black', ha='center',fontsize=14)

    if new_max_value!=None:
        # Add a horizontal dotted line for the maximum flow
        plt.axhline(y=new_max_value, color='brown', linestyle='--')
        # Add the label above the dotted line
        plt.text(0, (new_max_value+new_max_value*0.005), f'Link new Capacity', 
                 color='black', ha='center',fontsize=14)


    # Increase the size of the axis labels and title
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=20)

    # Increase tick size for both axes
    plt.xticks(x, plot_df[xlabel], fontsize=14)  # x-axis ticks
    plt.yticks(fontsize=14)                 # y-axis ticks

    # Add legend with increased fontsize
    plt.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{folder_path}/{file_name}_{title}_{date}.jpg', format='jpg', dpi=300)

def plot_one_bar(plot_df, barlabel1, xlabel, ylabel, title, date,
                    folder_path, file_name, color1="green"):
    """
    Plot one bar at a time.
    """
    plt.figure(figsize=(16, 8))
    x = np.arange(len(plot_df[xlabel]))

    bar1= plt.bar(x, plot_df[barlabel1], width=0.4, label=barlabel1,color=color1)
    # Add values in each bar
    for bar in bar1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, 
                yval/2, 
                round(yval, 2), 
                ha='center', 
                va='bottom',
                rotation=90,
                fontsize=12)

    # Increase the size of the axis labels and title
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=20)

    # Increase tick size for both axes
    plt.xticks(x, plot_df[xlabel], fontsize=14)  # x-axis ticks
    plt.yticks(fontsize=14)                 # y-axis ticks

    # Add legend with increased fontsize
    #plt.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{folder_path}/{file_name}_{title}_{date}.jpg', format='jpg', dpi=300)


    def custom_plot(self, x_column, y_column, plot_title, x_label, y_label, color='blue'):
        """
        Custom plot function that can be used for any kind of bar chart.
        :param x_column: Name of the column to use for the x-axis
        :param y_column: Name of the column to use for the y-axis
        :param plot_title: Title of the plot
        :param x_label: Label for the x-axis
        :param y_label: Label for the y-axis
        :param color: Color of the bars
        """
        plt.figure(figsize=(10, 6))
        plt.bar(self.df[x_column], self.df[y_column], color=color)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(plot_title)
        plt.xticks(rotation=45)
        plt.show()