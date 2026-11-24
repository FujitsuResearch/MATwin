#Module to add toll interventions in the links & visualise the effects in Folium map.

#Import required libraries
import numpy as np
import folium
from folium import FeatureGroup, LayerControl
from shapely.geometry import LineString
from shapely.geometry import Point
from branca.colormap import LinearColormap
import branca.colormap as cm
import pandas as pd
import geopandas as gpd
import json
import scipy.sparse as sps
import ast
import csv
from scipy.sparse import save_npz
import random
from pathlib import Path
import os
import sys
import copy
from functools import reduce
from mate.utils import read_paths, load_k_shortest_paths

# Set the CRS for now use EPSG:4326 for universal #use ESPG:2228 for Freshno;
network_crs = 'EPSG:4326'

from mate.etl import build_network
from pesuelogit.networks import read_OD
from mate.utils import load_k_shortest_paths
from mate.utils import write_paths


def get_links_indices(network, links_dict):
    """To return the indices of the links in the network.

    Args:
        network(Network): TransportationNetwork
        links_dict(Dict ):
    
    Returns:
        list_indices: List of links indices
    """

    list_indices=[]
    list_toll = list(network.links_dict.keys())

    for link, toll_price in links_dict.items():
        index = list_toll.index(link)
        list_indices.append(index)
      
    return list_indices

def fetch_all_ods_paths(network):
    """Creating Dict of ODpair: List of links(all as per shoortest paths). 

    Args:
        network(Network): TransportationNetwork
    
    Returns:
        od_paths_dict: Dictionary of OD pairs as key & paths as values.
    """ 
    od_paths_dict = {}
    lines = []
    total_paths = len(network.paths)
    for path, counter in zip(network.paths, range(total_paths)):
        line = []
        for node in path.nodes:
            line.append(node.key)
        lines.append(line)
        
    # Iterate over the paths in batches of 3 as each od pair has 3 paths
    for i in range(0, len(lines), 3):
        # Extract the current batch of 3 paths
        batch = lines[i:i+3]
    
        if len(batch) == 3:
            # Use the first and last elements of the first path as the key
            key = (batch[0][0], batch[0][-1])
            # Add the batch of 3 paths to the dictionary
            od_paths_dict[key] = batch
    return od_paths_dict


# new code for integration
class interventions:
    def __init__(self, network, links_df, nodes_df, folder_path, area_of_interest, _K, _K_short):
        """
        Initialize the class with required data to perform interventions.
        """
        self.network=network
        self.links_df=links_df
        self.nodes_df=nodes_df
        self.folder_path=folder_path
        self.area_of_interest=area_of_interest
        self._K=_K
        self._K_short=_K_short

    # Execute Toll intervention
    def execute_toll_intervention(self, dict_links_with_toll):

        #Adding toll interventions
        links_df, affected_od_pairs, toll_links_indices, common_toll_links_indices, common_toll_links_path_ids=add_interventions_by_toll(self.network, self.links_df, dict_links_with_toll, self._K)

        print(f'Toll intervention link indices are: {toll_links_indices}')
        print(f'common Closed link indices are: {common_toll_links_indices}')
        print(f'affected_od_pairs due to toll interventions: {affected_od_pairs}')
        print(f'Path ids having all toll interventions links are: {common_toll_links_path_ids}')

        #To generate _K_short shortest paths and write in disk

        if len(affected_od_pairs)>=1:
            new_paths = create_new_network_for_affected_ods(self.nodes_df,links_df,affected_od_pairs, common_toll_links_indices, 
                                                    self.area_of_interest, self._K_short, file_path=self.folder_path+f"/output/Q-sparse-{self.area_of_interest}-newpaths.npz")
        
        # add new paths in current paths of network.
        from mate.utils import write_paths
        if 'new_paths' in locals(): 
            write_paths(self.network.paths+ new_paths , filepath=self.folder_path+f'/output/paths-{self.area_of_interest}-k{self._K}-newpaths.csv')
        
        return links_df, affected_od_pairs, toll_links_indices, common_toll_links_indices, common_toll_links_path_ids

    def execute_road_closure(self, cap_change_factor, road_closure_links_dict):

        #Adding road closure interventions
        affected_od_pairs, closure_links_indices, common_closure_links_indices, common_closure_links_path_ids=add_interventions_by_road_closure(self.network, self.links_df, road_closure_links_dict,self._K)

        print(f'Closed link indices are: {closure_links_indices}')
        #print(f'common Closed link indices are: {common_closure_links_indices}')
        print(f'affected_od_pairs due to common closed links are: {affected_od_pairs}')
        print(f'Path ids having all common closed links are: {common_closure_links_path_ids}')


        #To generate K shortest paths and write in disk
        if len(affected_od_pairs)>=1:
            new_paths = create_new_network_for_affected_ods(self.nodes_df,self.links_df,affected_od_pairs, closure_links_indices, 
                                                self.area_of_interest, self._K_short, file_path=self.folder_path+f"/output/Q-sparse-{self.area_of_interest}-newpaths.npz")
    
        #Case-2 Replace all the paths of affected od pairs with the new paths & write the file.
        def replace_existing_paths():
            #Fetching all affected path indices
            affected_ods_paths_indexes=[]
            for path_index in common_closure_links_path_ids:
                affected_ods_paths_indexes.append([i for i in range(path_index, path_index + self._K)])

            affected_ods_paths_indexes_flt = [index for sublist in affected_ods_paths_indexes for index in sublist]

            paths_df=pd.read_csv(self.folder_path+f"/output/paths-{self.area_of_interest}-k{self._K}.csv", delimiter="\t", header=None)

            # Replace the new paths in network paths at the affected paths indices.
            import csv
            i=0
            for index in affected_ods_paths_indexes_flt:
                line = []
                for node in new_paths[i].nodes:
                        line.append(node.key)
                paths_df[0][index] = ','.join(map(str, line))
                i+=1

            with open(self.folder_path+f'/output/paths-{self.area_of_interest}-k{self._K}_replaced_newpaths.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                writer.writerows(paths_df.values.tolist())

        # For Full road closure
        if cap_change_factor==-1:
            if 'new_paths' in locals():
                replace_existing_paths() 
             
        # For Partial road closure
        elif (cap_change_factor>-1) & (cap_change_factor<0):
            # Case-1 Add the new paths for the affected OD pairs along with keeping the old paths.
            # write file after adding the new paths in current paths of network.
            from mate.utils import write_paths
            if 'new_paths' in locals():
                write_paths(self.network.paths + new_paths , filepath=self.folder_path+f'/output/paths-{self.area_of_interest}-k{self._K}-old-newpaths.csv')
        
        return affected_od_pairs, closure_links_indices, common_closure_links_indices, common_closure_links_path_ids

    def execute_road_enhancement(self, cap_change_factor, cap_enhancement_links_dict):
        
        # Adding road capacity enhancement interventions
        cap_enhanced_links_indices = add_interventions_by_road_enhancement(self.network, self.links_df, cap_enhancement_links_dict,self._K)

        print(f'Cap enhanced link indices are: {cap_enhanced_links_indices}')
        print(f' OD pairs and paths: ({len(self.network.ods)}, {len(self.network.paths)})')

        return cap_enhanced_links_indices

def add_interventions_by_toll(network, links_df, toll_links_dict, _K):
    """To add tolls intervention to the network. 

    Args:
        network(Network): TransportationNetwork
        links_df(pandas.DataFrame): links data of network.
        toll_links_dict(Dict):toll links dictionary.Format: {(link key): toll price})
    
    Returns:
        closure_links_indices: closure links index in network.
        links_df: Links dataframe with toll data
    """ 
    
    # Missing element to add to each toll links as data in network is present in this format.
    element_to_add = '0'
    # Create a new dictionary with updated keys
    toll_links_dict_upd = {(key + (element_to_add,)): value for key, value in toll_links_dict.items()}
    # Get the indices of the closure links in the network.
    toll_links_indices = get_links_indices(network, toll_links_dict_upd)
    print(f'Toll link indices are: {toll_links_indices}')
    
    affected_ods_list, common_toll_link_ids_list, toll_links_path_ids =check_common_toll_links(network, toll_links_indices, _K)
    
     # Update 'links_df' with 'toll_price' using the map function
    links_df = links_df.assign(toll_price=links_df.link_key.map(lambda x: toll_links_dict_upd.get(x, 0)))

    return links_df, affected_ods_list, toll_links_indices, common_toll_link_ids_list, toll_links_path_ids

#Function to check if toll links are common in all paths for any od
def check_common_toll_links(network, toll_links_indices, _K):

    # Fetch all paths dict for each od
    od_path_links_dict=fetch_od_paths_dict(network)

    # Fetch all the ODs with all common links in all its paths
    od_path_id_list, ods_common_links_list= fetch_od_common_links(od_path_links_dict, _K)
    
    affected_ods_list=[]
    common_toll_link_ids_list=[]
    toll_links_path_id=[]
    
    for toll_link_id in toll_links_indices:
        for i in range(len(ods_common_links_list)):
            if toll_link_id in [common_link_id for common_link_id in ods_common_links_list[i]]:
                init_node=network.paths[od_path_id_list[i]].nodes[0].key
                term_node=network.paths[od_path_id_list[i]].nodes[-1].key
                od_pair=(init_node, term_node)
                if od_pair not in affected_ods_list:
                    affected_ods_list.append(od_pair)
                    toll_links_path_id.append(od_path_id_list[i])
                    common_toll_link_ids_list.append(toll_link_id)

    return affected_ods_list, list(set(common_toll_link_ids_list)), toll_links_path_id

def fetch_od_paths_dict(network):
    D_matrix=network.D # link*path matrix
    # Fetch the link ids existing for each od path
    row_indices, col_indices = np.where(D_matrix.transpose() == 1)
    row_col_dict = {}
    for row, col in zip(row_indices, col_indices):
        if row not in row_col_dict:
            row_col_dict[row] = []
        row_col_dict[row].append(col)
    return row_col_dict


def fetch_od_common_links(od_path_links_dict, _K):
    
    # Initialize an empty list to hold the common link ids in each path
    common_links_ids_list = []

    # To hold the starting index of each path
    path_id_list=[]

    # Convert the dictionary to a list of tuples (key, value)
    items = list(od_path_links_dict.items())

    # Iterate over the dictionary items in steps of 3 as there are 3 paths for each OD
    for i in range(0, len(items), _K):
        if i + (_K-1) < len(items):
            # Take 3 consecutive key-value pairs
            group = items[i:i + _K]
        
            # Extract the lists
            lists = [value for key, value in group]
        
            # Find common elements in all paths of each OD
            common_elements = reduce(set.intersection, map(set, lists))
            if len(common_elements)>0:
                # Add the common elements to the final list
                common_links_ids_list.append(list(common_elements))
                path_id_list.append(i)
    return path_id_list, common_links_ids_list

def add_interventions_by_road_closure(network, links_df, closure_links_dict, _K):
    """To add road closure intervention to the network. 

    Args:
        network(Network): TransportationNetwork
        links_df(pandas.DataFrame): links data of network.
        closure_links_dict(Dict):Closure links dictionary.Format: {(link key): toll price})
    
    Returns:
        closure_links_indices: closure links index in network.
        links_df: Links dataframe with toll data
    """ 
    
    # Missing element to add to each toll links as data in network is present in this format.
    element_to_add = '0'
    # Create a new dictionary with updated keys
    closure_links_dict_upd = {(key + (element_to_add,)): value for key, value in closure_links_dict.items()}
    # Get the indices of the closure links in the network.
    closure_links_indices = get_links_indices(network, closure_links_dict_upd)
    print(f'Closure link indices are: {closure_links_indices}')
    
    affected_ods_list, common_closure_link_ids_list, closure_links_path_ids =check_common_closure_links(network, closure_links_indices,_K)
    return affected_ods_list, closure_links_indices, common_closure_link_ids_list, closure_links_path_ids

#Function to check if closure links are common in all paths for any od
def check_common_closure_links(network, closure_links_indices,_K):

    # Fetch all paths dict for search od
    od_path_links_dict=fetch_od_paths_dict(network)

    # Fetch all the ODs with all used linksin its all paths
    od_path_id_list, ods_used_links_list= fetch_od_all_links(od_path_links_dict,_K)
    
    affected_ods_list=[]
    common_closure_link_ids_list=[]
    closure_links_path_id=[]
    
    for closure_link_id in closure_links_indices:
        for i in range(len(ods_used_links_list)):
            if closure_link_id in [common_link_id for common_link_id in ods_used_links_list[i]]:
                init_node=network.paths[od_path_id_list[i]].nodes[0].key
                term_node=network.paths[od_path_id_list[i]].nodes[-1].key
                od_pair=(init_node, term_node)
                if od_pair not in affected_ods_list:
                    affected_ods_list.append(od_pair)
                    closure_links_path_id.append(od_path_id_list[i])
                    common_closure_link_ids_list.append(closure_link_id)

    return affected_ods_list, list(set(common_closure_link_ids_list)), closure_links_path_id

def fetch_od_all_links(od_path_links_dict,_K):
    
    # Initialize an empty list to hold the common link ids in each path
    all_nodes_ids_list = []

    # To hold the starting index of each path
    path_id_list=[]

    # Convert the dictionary to a list of tuples (key, value)
    items = list(od_path_links_dict.items())

    # Iterate over the dictionary items in steps of paths for each OD pair
    for i in range(0, len(items), _K):
        if i + (_K-1) < len(items):
            # Take 3 consecutive key-value pairs
            group = items[i:i+_K]
        
            # Extract the lists
            lists = [value for key, value in group]
        
            # Fetch all the elements/nodes in all paths of a OD pair.
            common_elements = reduce(set.union, map(set, lists))
            if len(common_elements)>0:
                # Add the common elements to the final list
                all_nodes_ids_list.append(list(common_elements))
                path_id_list.append(i)
    return path_id_list, all_nodes_ids_list

def analyse_tf_tt(links_df, bf_flow, af_flow, bf_tt, af_tt):
    """To analyse the effect of intervention on traffic flow & travel time

    """

    changes_in_links = links_df.copy(deep=True)
    changes_in_links['flow_change'] = (af_flow - bf_flow)[0]
    changes_in_links['tt_change'] = (af_tt - bf_tt)[0] 
    top_flow_changes=changes_in_links.sort_values(by=['flow_change', 'tt_change'], 
                                 ascending=False)
    least_flow_changes=changes_in_links.sort_values(by=['flow_change', 'tt_change'], 
                                 ascending=True)
    
    positive_flow_change = changes_in_links[changes_in_links.flow_change>0][['link_key', 'u', 'v', 
                                                                             'length','link_type', 
                                                                             'toll_price','flow_change', 
                                                                             'tt_change']].sort_values(by=['flow_change'], ascending=False)

    negative_flow_change = changes_in_links[changes_in_links.flow_change<0][['link_key', 'u', 'v', 
                                                                             'length','link_type', 
                                                                             'toll_price','flow_change', 
                                                                             'tt_change']].sort_values(by=['flow_change'], ascending=True)

    print(f'Major changes in traffic flow & travel time:  {top_flow_changes}')
    print(f'Minor changes in traffic flow & travel time:  {least_flow_changes}')
    print(f'Positive traffic flow changes: {positive_flow_change}')
    print(f'Negative traffic flow changes: {negative_flow_change}')


def add_interventions_by_road_enhancement(network, links_df, enhance_links_dict, _K):
    """To add road enhancement intervention to the network. 

    Args:
        network(Network): TransportationNetwork
        links_df(pandas.DataFrame): links data of network.
        enhanced_links_dict(Dict):Enhanced links dictionary.Format: {(link key): toll price})
    
    Returns:
        enhanced_links_indices: enhanced links index in network.
        links_df: Links dataframe with toll data
    """ 
    
    # Missing element to add to each toll links as data in network is present in this format.
    element_to_add = '0'
    # Create a new dictionary with updated keys
    enhanced_links_dict_upd = {(key + (element_to_add,)): value for key, value in enhance_links_dict.items()}
    # Get the indices of the enhanced links in the network.
    enhanced_links_indices = get_links_indices(network, enhanced_links_dict_upd)
    print(f'enhanced link indices are: {enhanced_links_indices}')
    
    return enhanced_links_indices


# Function : Path to traverse target link
# This function is not in use for now. Will be optimized later when this functionality is in use.
#gdf_taz_input, link_specific_input,links_data_input, node_data_input,q-sparse-fresno-with-target-link_input,paths-fresno-k{_K}_target_link_output,paths-fresno-{m}-target-link_output)
# def paths_to_travese_target_links():
#     # Read the TAZ input files
#     gdf_taz = gpd.read_file('./examples/data_pre_processing/data/input/TAZ_Shap_Files/tz06_d00_shp/tz06_d00.shp')

#     # Get the link specific data 
#     link_specific_data = pd.read_csv("./examples/data_pre_processing/data/output/link_specific_data.csv", sep='\t')

#     # Get one target link 
#     real_links = pd.read_csv("./examples/data_pre_processing/data/output/links_data.csv")
#     o,u,v = real_links[real_links.highway == 'primary'][['osmid', 'u','v']].iloc[0]

#     target_link_info = link_specific_data[(link_specific_data.u == u) & (link_specific_data.v ==v)]

#     # Create a LineString of the link 
#     target_link = LineString([(target_link_info.lon_u, target_link_info.lat_u), (target_link_info.lon_v, target_link_info.lat_v)])

#     # Find the TAZ in which the target link lies 
#     for taz in gdf_taz.geometry:
#         if taz.contains(target_link):
#             print(f"The link lies in {taz}")
#             break
#     else:
#         print("The link does not lie within any of the zone.")

#     # Read the nodes
#     nodes_df = pd.read_csv("./examples/data_pre_processing/data/output/node_specific_data.csv", sep="\t")

#     # Read the links
#     links_df = pd.read_csv(
#         "./examples/data_pre_processing/data/output/link_specific_data.csv",
#         sep="\t",
#         converters={"link_key": ast.literal_eval},
#     )

#     rows, cols = nodes_df.shape[0], nodes_df.shape[0]

#     # create an empty matrix
#     sps_acc = sps.coo_matrix((rows, cols))

#     whole_mat = np.zeros((nodes_df.shape[0], nodes_df.shape[0]))
#     gen_shape = nodes_df[nodes_df["type"].isin(["generator"])].shape[0]
#     mat_for_gen = np.ones((gen_shape, gen_shape)) - np.diag(np.ones(gen_shape))
#     index_attractor, index_generator = nodes_df[nodes_df["type"].isin(["attractor"])].index[0], nodes_df[nodes_df["type"].isin(["generator"])].index[0]

#     index_u = nodes_df[(nodes_df['lon'] == target_link.coords[0][0]) & (nodes_df['lat'] == target_link.coords[0][1])].index

#     index_v = nodes_df[(nodes_df['lon'] == target_link.coords[1][0]) & (nodes_df['lat'] == target_link.coords[1][1])].index

#     # Check the value of u, v before setting the value 
#     print(f"The value of u, v pre-assignment is: ", whole_mat[index_u[0]][index_v[0]])
#     # Set the path from u to v (target link) by assigning 1
#     # whole_mat[index_u[0]][index_v[0]] = 1.0
#     print(f"The value of u, v post-assignment is : ", whole_mat[index_u[0]][index_v[0]])

#     # Indices of all generator 
#     indices_generator = nodes_df[nodes_df["type"].isin(["generator"])].index

#     gen_nodes = nodes_df[nodes_df["type"].isin(["generator"])]
#     vec_gen_to_u_node = np.ones((gen_nodes.shape[0], index_u.shape[0]))

#     # Set all the paths from generator to the u node 
#     whole_mat[indices_generator[0]:indices_generator[-1]+1, index_u[0]:index_u[0]+1] = vec_gen_to_u_node

#     # Indices of all attractor 
#     indices_attractor = nodes_df[nodes_df["type"].isin(["attractor"])].index

#     att_nodes = nodes_df[nodes_df["type"].isin(["attractor"])]
#     vec_v_node_to_attractor = np.ones((index_v.shape[0], att_nodes.shape[0]))

#     whole_mat[index_v[0]:index_v[0]+1, indices_attractor[0]:indices_attractor[-1]+1] = vec_v_node_to_attractor

#     mat_for_gen = np.ones((gen_shape, gen_shape)) - np.diag(np.ones(gen_shape))
#     whole_mat[index_generator:, index_attractor:index_generator] = mat_for_gen

#     # Create the coo_matrix
#     sps_acc = sps.coo_matrix((whole_mat), shape=whole_mat.shape)

#     # Save the sparse matrix to a .npz file
#     save_npz("examples/data_pre_processing/data/output/Q-sparse-fresno-with-target-link.npz", sps_acc)

#     nodes_df = pd.read_csv("./examples/data_pre_processing/data/output/node_specific_data.csv", sep="\t")

#     links_df = pd.read_csv(
#         "./examples/data_pre_processing/data/output/link_specific_data.csv",
#         sep="\t",
#         converters={"link_key": ast.literal_eval},  # , "pems_id": ast.literal_eval
#     )

#     # build the network 
#     network = build_network(
#         links_df=links_df,
#         nodes_df=nodes_df,
#         node_id_label="osmid",
#         link_id_label="osmid_link",
#         crs=network_crs,
#         key="fresno",
#     )

#     #read OD matrix 
#     read_OD(
#         network=network,
#         sparse=True,
#         filepath="examples/data_pre_processing/data/output/Q-sparse-fresno-with-target-link.npz",
#     )
#     _K = 3
#     load_k_shortest_paths(network=network, k=_K, update_incidence_matrices=True, block_output=False)
#     # Write the paths
#     write_paths(network.paths, filepath=f'./examples/data_pre_processing/data/output/paths-fresno-k{_K}_target_link.csv')

#     list_of_path_end_with_u = []
#     list_of_path_start_with_v = []

#     # Read the file with paths including the target link
#     with open(f'./examples/data_pre_processing/data/output/paths-fresno-k{_K}_target_link.csv', 'r') as file:
#         reader = csv.reader(file)
#         for row in reader:
#             if row[-1] ==  str(index_u[0]) : #Loop over paths to search for all paths that have 4 (u) as destination
#                 list_of_path_end_with_u.append(row)
#             elif row[0] == str(index_v[0]): #Loop over paths to search for all paths that have 5 (v) as origin
#                 list_of_path_start_with_v.append(row)

#     # Initialize an empty list to hold the combined elements
#     combined_path_u_v_list = []

#     for i in range(len(list_of_path_end_with_u)):
#         for j in range (len(list_of_path_start_with_v)):
#             # combine each row from each of the list, and then apped to combined_path_u_v_list list
#             combined_path_u_v_list.append(list_of_path_end_with_u[i] + list_of_path_start_with_v[j])

#     # typecast the values in paths from strng to numpy int for comparison
#     list_int_converted_combined_links = [[np.int64(s) for s in sublist] for sublist in combined_path_u_v_list] 
#     len(list_int_converted_combined_links)

#     #Get the paths which match the OD pairs or else we will face issue in validation as it results in OD pair mismatch 

#     selected_path_u_v__matched_in_OD_list = []

#     for path in list_int_converted_combined_links:
#         o,d = path[0], path[-1]
#         if (o,d) in network.ods:
#             #print("Matching od pair ", o,d)
#             selected_path_u_v__matched_in_OD_list.append(path)

#     len(selected_path_u_v__matched_in_OD_list)

#     m = 100
#     selected_paths = random.sample(selected_path_u_v__matched_in_OD_list, m)
#     #Typecast to string
#     list_100_selected_paths = [[str(s) for s in sublist] for sublist in selected_paths] 

#     filepath = f'./examples/data_pre_processing/data/output/paths-fresno-{m}-target-link.csv'
#     with open(filepath, 'w', newline='') as csvfile:
#             writer = csv.writer(csvfile, delimiter=',')
#             writer.writerows(list_100_selected_paths)

#     return

def paths_to_travese_target_links(link_traverse, m=10, tf=-1, model_fixed_effect=None):
    
    if model_fixed_effect is None:
        model_fixed_effect= np.zeros(links_df.shape[0])
        
    target_link_info = links_df[(links_df.u == link_traverse[0]) & (links_df.v == link_traverse[1])]
    gen_shape = nodes_df[nodes_df["type"].isin(["generator"])].shape[0]
    
    index_attractor, index_generator = nodes_df[nodes_df["type"].isin(["attractor"])].index[0], nodes_df[nodes_df["type"].isin(["generator"])].index[0]
    
    whole_mat = np.zeros((nodes_df.shape[0], nodes_df.shape[0]))
    index_u, index_v = link_traverse[0], link_traverse[1]
    whole_mat[index_u, index_v] = 1.0
    
    # Indices of all generator 
    indices_generator = nodes_df[nodes_df["type"].isin(["generator"])].index
    
    gen_nodes = nodes_df[nodes_df["type"].isin(["generator"])]
    vec_gen_to_u_node = np.ones((gen_nodes.shape[0], 1))
    
    # Set all the paths from generator to the u node 
    whole_mat[indices_generator[0]:indices_generator[-1]+1, index_u:index_u+1] = vec_gen_to_u_node
    
    # Indices of all attractor 
    indices_attractor = nodes_df[nodes_df["type"].isin(["attractor"])].index
    
    att_nodes = nodes_df[nodes_df["type"].isin(["attractor"])]
    vec_v_node_to_attractor = np.ones((1, att_nodes.shape[0]))
    
    whole_mat[index_v:index_v+1, indices_attractor[0]:indices_attractor[-1]+1] = vec_v_node_to_attractor
    
    mat_for_gen = np.ones((gen_shape, gen_shape)) - np.diag(np.ones(gen_shape))
    whole_mat[index_generator:, index_attractor:index_generator] = mat_for_gen
    
    # Create the coo_matrix
    sps_acc = sps.coo_matrix((whole_mat), shape=whole_mat.shape)
    file_path = folder_path+f"/output/Q-sparse-{area_of_interest}-with-target-link.npz"
    # Save the sparse matrix to a .npz file
    save_npz(file_path, sps_acc)
    
    network = build_network(
        links_df=links_df,
        nodes_df=nodes_df,
        node_id_label="osmid",
        link_id_label="osmid_link",
        crs=network_crs,
        key=area_of_interest,
    )
    
    #read OD matrix 
    read_OD(
        network=network,
        sparse=True,
        filepath=file_path,
    )
    _K = 3
    load_k_shortest_paths(network=network, k=_K, update_incidence_matrices=True, block_output=False)

    link_n = {}
    for k, v in links_df.link_key.to_dict().items():
        link_n[v] = k
    
    whole_od_paths = {}
    for od, paths in network.paths_od.items():
        path_utility, path_tt = 0, 0
        for path in paths:
            for link in path.links:
                path_utility += link.utility({'tf':tf}) + model_fixed_effect[link_n[link.key]]
                path_tt += link.traveltime
        whole_od_paths[od] = [path_utility/len(paths), path_tt/len(paths)]
    
    created_paths = []
    for path in network.paths:
        path_nodes = []
        path_utility, path_tt = 0, 0
    
        for link in path.links:
            # print(link.init_node.key, link.term_node.key, end=' ')
            path_utility += link.utility({'tf':tf}) + model_fixed_effect[link_n[link.key]]
            path_tt += link.traveltime
            path_nodes.append(link.init_node.key)
        path_nodes.append(link.term_node.key)
        od = [path_nodes[0], path_nodes[-1]]
        created_paths.append([od, path_nodes, path_utility, path_tt])
    
    list_of_path_end_with_u = []
    list_of_path_start_with_v = []
    
    for path in created_paths:
        if path[0][1] == index_u:
            list_of_path_end_with_u.append(path)
        elif path[0][0] == index_v:
            list_of_path_start_with_v.append(path)
    
    # Initialize an empty list to hold the combined elements
    combined_path_u_v_list = []
    for u_end in list_of_path_end_with_u:
        for v_start in list_of_path_start_with_v:
            # combine each row from each of the list, and then apped to combined_path_u_v_list list
            od = [u_end[0][0], v_start[0][1]]
            path = u_end[1] + v_start[1]
            path_utility = u_end[2] + v_start[2]
            path_tt = u_end[3] + v_start[3]
            combined_path_u_v_list.append([od, path, path_utility, path_tt])
    
    print("all combined paths length", len(combined_path_u_v_list))
    
    selected_path_u_v__matched_in_OD_list = []
    for path in combined_path_u_v_list:
        o,d = path[0]
        if (o,d) in network.ods:
            #print("Matching od pair ", o,d)
            selected_path_u_v__matched_in_OD_list.append(path)
    
    print("selected with matched od path lengths", len(selected_path_u_v__matched_in_OD_list))
    
    
    similar_paths = []
    
    for paths in selected_path_u_v__matched_in_OD_list:
        od, path, utility, tt = paths
        actual_utility, actual_tt = whole_od_paths[(od[0], od[1])]

        if (utility >= actual_utility) and (tt <=actual_tt):
            paths.extend([(utility-actual_utility)/actual_utility + (tt-actual_tt)/actual_tt])
            similar_paths.append(paths)
            # print(actual_utility, utility, actual_tt, tt)
        # if (.95*actual_utility>= utility >= 1.05*actual_utility) or (.95*actual_tt<= tt <= 1.05*actual_tt):
        #     similar_paths.append(paths)
    
    print("similar paths found", len(similar_paths))
    sorted_path = sorted(similar_paths, key=lambda lst: lst[-1], reverse=False)

    selected_paths = sorted_path[:m]
    # selected_paths = random.sample(similar_paths, m)
    selected_paths = [i[1] for i in selected_paths]
    return selected_paths


def create_new_network_for_affected_ods(nodes_df, links_df,affected_od_pairs, common_closure_links_indices,area_of_interest, _K,file_path):

    new_network = build_network(
    links_df=links_df,
    nodes_df=nodes_df,
    node_id_label="osmid",
    link_id_label="osmid_link",
    crs=network_crs,
    key=area_of_interest,
    )   
    #new_network=copy.deepcopy(network,None, None)
    # Creating a blank network as per number of nodes.
    rows, cols = nodes_df.shape[0], nodes_df.shape[0]
    # create an empty matrix
    sps_acc = sps.coo_matrix((rows, cols))

    # Creating a OD trips only for affected od pairs.
    od_whole_mat = np.zeros((nodes_df.shape[0], nodes_df.shape[0]))
    for od_init, od_term in affected_od_pairs:
        od_whole_mat[(od_init, od_term)] = 1
    
    # Create the coo_matrix
    sps_acc = sps.coo_matrix((od_whole_mat), shape=od_whole_mat.shape)
    # Save the sparse matrix to a .npz file
    save_npz(file_path, sps_acc)

    read_OD(
    network=new_network,
    sparse=True,
    filepath=file_path,
    )   
    
    # Fecth the existing Adjacency matrix & marking 0 values to the common closure links.
    init_nodes = []
    term_nodes = []

    # For the closure links, marking them in adjacency matrix of network as 0 as they are not of use.
    for common_closure_link in links_df.iloc[common_closure_links_indices]["link_key"].values:
        init_nodes.append(common_closure_link[0])
        term_nodes.append(common_closure_link[1])
    
    for init_node, term_node in zip(init_nodes,term_nodes):
        new_network.A[(int(init_node), int(term_node))] = 0
    
    theta_utility ={'tf':-1} #using free-flow travel time 
    load_k_shortest_paths(network=new_network, theta=theta_utility, k=_K, update_incidence_matrices=True, block_output=False)
    return new_network.paths