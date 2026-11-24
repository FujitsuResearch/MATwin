import os
import requests
import json
import gzip
import osmnx as ox
import math
import ast
from shapely import geometry
from shapely.geometry import box
from shapely.geometry import LineString
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
import math
from math import atan2, cos, sin, degrees
#import cenpy
import numpy as np
from census import Census
import zipfile, io
import re
import random
from sklearn.impute import SimpleImputer
import scipy.sparse as sps
from scipy.sparse import save_npz

# Username and password for scrapping data from pems site
pems_payload = {
    "username": os.environ.get("PEMS_USERNAME", ""),
    "password": os.environ.get("PEMS_PASSWORD", "")
}

# Api key for census data site: https://pygis.io/docs/d_access_census.html
API_KEY = os.environ.get("CENSUS_API_KEY", "")

network_crs = 'EPSG:4326'


def _credential(name, value=""):
    value = value or os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set.")
    return value


def _pems_listing(response):
    try:
        return json.loads(response.content)
    except ValueError:
        raise RuntimeError("PeMS Clearinghouse did not return a file listing "
                           "(login failed or the service is unavailable).") from None


def show_hour_data(hour_data_json: json):
    option_lst = []
    print("{0:2} -> {1:12} {2:40} {3:25}".format('id', 'key', 'file_name', 'url'))
    for n, (key, val) in enumerate(hour_data_json['data'].items(), 1):
        print('{0:2} -> {1:12} {2:40} {3:25}'.format(n, key, val[0]['file_name'], val[0]['url']))
        option_lst.append([val[0]['file_name'], val[0]['url']])
    return option_lst

def show_meta_data(meta_data_json: json, format='text'):
    option_lst = []
    print('{0:5} {1:12} {2:12} {3:12} {4:40} {5:25}'.format('id','key', 'from_day', 'to_day', 'file_name', 'url'))
    #print('id', 'key', 'from_day', 'to_day', 'file_name', 'url')
    #print("Meta data :: ",meta_data_json['data'].items())
    for n, (key, val) in enumerate(meta_data_json['data'].items(), 1):
        print('{0:2} -> {1:12} {2:12} {3:12} {4:40} {5:25}'.format(n, key, val['from_day'], val['to_day'], val['format'][format]['file_name'], val['format'][format]['url']))
        option_lst.append([val['format'][format]['file_name'], val['format'][format]['url']])
    return option_lst

def scrape_pems_data(file_m_option,
                    file_h_option,
                    req_format=1,
                    year1=2019,
                    year2 = 2019, 
                    district_id=6, 
                    dir="./examples/notebooks/data", 
                    payload=pems_payload, 

                    ):
    """Downloads required year's data for given district and unzips it to the provided dir.

    Args:
        req_format (int, optional): File format to be downloaded. Defaults to 1(text). Supported File formats are either text or tmdd.
        year1 (int, optional): Year for the station hour data. Defaults to 2019.
        year2 (int, optional): Year for the station metadata. Defaults to 2019.
        district_id (int, optional): District ID. Defaults to 6.
        dir (str, optional): Directory to save downloaded data. Defaults to "./examples/notebooks/data/input/".
        payload (dict, optional): Dict of username-password for pems site. Defaults to pems_payload.  
    """    
    prefix_download_url = 'https://pems.dot.ca.gov'
    site_url = "https://pems.dot.ca.gov/?dnode=Clearinghouse"
    
    # # >> station hour data url
    station_hour_url = f"https://pems.dot.ca.gov/?srq=clearinghouse&district_id={district_id}&geotag=&yy={year1}&type=station_hour&returnformat=text"
    station_metadata_url = f"https://pems.dot.ca.gov/?srq=clearinghouse&district_id={district_id}&geotag=&yy={year2}&type=metadata&returnformat=metadata"
    
    # Check if dir exists if not create it
    if not os.path.exists(dir):
        print("creating directory at {0}".format(dir))
        #os.mkdir(dir)
        os.makedirs(dir, exist_ok=True)
        os.mkdir(os.path.join(dir, 'input'))
        os.mkdir(os.path.join(dir, 'output'))
        os.mkdir(os.path.join(dir, 'output/area_expansion'))
        print("Note: Add input files at {0}".format(os.path.join(dir, 'input')))
    dir = os.path.join(dir, "input")
    payload = {"username": _credential("PEMS_USERNAME", payload.get("username")),
               "password": _credential("PEMS_PASSWORD", payload.get("password"))}
    with requests.session() as s:
        s.post(site_url, data=payload)
        
        # # >> station metadata
        print("Station meta data\n")
        r_meta = s.get(station_metadata_url)
        meta_data_json = _pems_listing(r_meta)
    
        # req_format = input("please select format\n1. text\n2. tmdd\n")
        # req_format = 'text' if req_format == '1' else 'tmdd'
        # print(req_format)
        
        meta_op_list = show_meta_data(meta_data_json, req_format)
        # file_m_option = input("please select file to download")
        # print(file_m_option)
        
        # print(meta_op_list)
        # download the station metadata
        down_meta_url = prefix_download_url + meta_op_list[int(file_m_option) - 1][1]
        file = s.get(down_meta_url, stream=True)
        file_location = os.path.join(dir, meta_op_list[int(file_m_option) - 1][0])
        with open(file_location, 'wb') as f:
            for data in (file.iter_content()):
                f.write(data)
        print(meta_op_list[int(file_m_option) - 1][0], 'file downloaded\n')
    
        # # >> station hour data
        r_hour = s.get(station_hour_url)
        hour_data_json = _pems_listing(r_hour)
        
        # print("Station hour data\n")
        hour_data_list = show_hour_data(hour_data_json)
        # file_h_option = input("please select id of file to download")
        
        down_hour_url = prefix_download_url + hour_data_list[int(file_h_option) - 1][1]
        file = s.get(down_hour_url, stream=True)
        file_location = os.path.join(dir, hour_data_list[int(file_h_option) - 1][0])
        with open(file_location, 'wb') as f:
            for data in (file.iter_content()):
                f.write(data)
    
        print(hour_data_list[int(file_h_option) - 1][0], 'file downloaded')
    
    infile = os.path.join(dir, hour_data_list[int(file_h_option) - 1][0])
    tofile = os.path.join(dir, hour_data_list[int(file_h_option) - 1][0][:-3])

    with open(infile, 'rb') as inf, open(tofile, 'w', encoding='utf8') as tof:
        decom_str = gzip.decompress(inf.read()).decode('utf-8')
        tof.write(decom_str)
    

class BoundingBox(object):
    def __init__(self, *args, **kwargs):
        self.lat_min = None
        self.lon_min = None
        self.lat_max = None
        self.lon_max = None


def get_bounding_box(latitude_in_degrees, longitude_in_degrees, width_km, height_km):
    assert width_km > 0 and height_km > 0
    assert latitude_in_degrees >= -90.0 and latitude_in_degrees  <= 90.0
    assert longitude_in_degrees >= -180.0 and longitude_in_degrees <= 180.0

    # half_side_in_km = half_side_in_miles * 1.609344
    lat = math.radians(latitude_in_degrees)
    lon = math.radians(longitude_in_degrees)

    radius  = 6371
    # Radius of the parallel at given latitude
    parallel_radius = radius*math.cos(lat)

    lat_min = lat - (height_km/2)/radius
    lat_max = lat + (height_km/2)/radius
    lon_min = lon - (width_km/2)/parallel_radius
    lon_max = lon + (width_km/2)/parallel_radius
    rad2deg = math.degrees

    box = BoundingBox()
    box.lat_min = rad2deg(lat_min)
    box.lon_min = rad2deg(lon_min)
    box.lat_max = rad2deg(lat_max)
    box.lon_max = rad2deg(lon_max)

    return (box)

def create_polygon_from_bbox(bbox):
    p1 = geometry.Point(bbox.lon_max, bbox.lat_max)
    p2 = geometry.Point(bbox.lon_min, bbox.lat_max)
    p3 = geometry.Point(bbox.lon_min, bbox.lat_min)
    p4 = geometry.Point(bbox.lon_max, bbox.lat_min)
    
    pointList = [p1, p2, p3, p4, p1]
    
    poly = geometry.Polygon(pointList)
    return poly


def lane_transformed_to_int(i):
    try:
        if isinstance(i, int):
            return i
        else:
            val = ast.literal_eval(i)
            if isinstance(val, int):
                return val
            elif isinstance(val, list):
                return max(list(map(int, val)))
            else:
                return 1
    except:
        return 1

def retrieve_graph_from_osm(area_of_interest,
                            lat, 
                            lon, 
                            width_km=2,
                            height_km=5, 
                            links_file="links_data.csv",
                            nodes_file="nodes_data.csv",
                            links_df_file="link_list.geojson",
                            area_gdf_file="selected_area_gdf.shp",
                            dir="./examples/notebooks/data/output"):
    """Gets open street map data for bounding box and creates files for links, nodes and selected bounding box.

    Args:
        area_of_interest (str) : City name for which we want to download the data.
        lat (float): Latitude of center point for defined bounding box.
        lon (float): Longitude of center point for defined bounding box.
        width_km (float, optional): Width of Bounding Box. Defaults to 2.
        height_km (float, optional): Height of Bounding Box. Defaults to 5.
        links_file (str, optional): Filename for storing real links data to csv. Defaults to "links_data.csv".
        nodes_file (str, optional): Filename for storing real nodes data to csv. Defaults to "nodes_data.csv".
        links_df_file (str, optional): Filename for storing real links data to geodata frame. Defaults to "link_list.geojson".
        area_gdf_file (str, optional): Filename for storing bounding box data to geodata frame. Defaults to "selected_area_gdf.shp".
        dir (str, optional): Directory to save downloaded data. Defaults to "./examples/notebooks/data/output".
    """    
    bbox = get_bounding_box(latitude_in_degrees=lat,longitude_in_degrees=lon, width_km=width_km, height_km=height_km)
    polygon = create_polygon_from_bbox(bbox)
    print("Polygon", polygon)
    #graph = ox.graph_from_polygon(polygon, simplify=True, truncate_by_edge=True)
    graph = ox.graph_from_polygon(polygon, simplify=True, truncate_by_edge=True, network_type='drive')
    
    nodes, links = ox.graph_to_gdfs(graph)
    links.reset_index(inplace=True)
    links = links[~links.duplicated(subset=['u', 'v'], keep='last')]
    print("Total links:{0} and nodes:{1} in selected bounding box".format(links.shape, nodes.shape))
    
    graph = ox.utils_graph.get_largest_component(graph, strongly=True)
    nodes, links = ox.graph_to_gdfs(graph)
    links.reset_index(inplace=True)
    links = links[~links.duplicated(subset=['u', 'v'], keep='last')]
    print("Strongly connected links:{0} nodes:{1} in selected bounding box".format(links.shape, nodes.shape))

    links.lanes.fillna(1, inplace=True)
    links.lanes = links.lanes.apply(lane_transformed_to_int)
    links['capacity'] = 2000 * links.lanes 
    
    # Save links data to csv
    links.to_csv(os.path.join(dir, links_file), sep=',', index=False, columns=['osmid', 'u', 'v', 'length', 'highway', 'maxspeed', 'lanes', 'capacity'])
    
    # Save nodes data to csv
    nodes.reset_index(inplace=True)
    nodes.to_csv(os.path.join(dir, nodes_file), sep=',', index=False, columns=["osmid", 'x', 'y'])

    links_df = links[['osmid', 'u', 'v', 'length', 'highway', 'maxspeed', 'lanes', 'capacity', 'geometry']]
    
    with open(os.path.join(dir, links_df_file), "w") as f:
        json.dump(links_df.__geo_interface__, f) 
    
    # Save selected area in geodata frame
    gdf = gpd.geoseries.from_wkt([polygon.wkt], crs='EPSG:4326')
    gdf= gpd.GeoDataFrame(data={"city": [area_of_interest]}, geometry=[polygon])
    gdf.to_file(os.path.join(dir, area_gdf_file), crs='EPSG:4326')
    

def traffic_counter_to_link(station_metadata_file = "input/d06_text_meta_2019_04_20.txt",
                            area_gdf_file = "output/selected_area_gdf.shp",
                            nodes_file = "output/nodes_data.csv",
                            links_df_file = "output/link_list.geojson",
                            dist_threshold = 30,
                            counter_link_joined_data_file = "output/station_links_joined_data.geojson",
                            dir="./examples/notebooks/data/"):
    """Maps traffic counter/loop detecter to network links.

    Args:
        station_metadata_file (str, optional): Station metadata from pems. Defaults to "input/d06_text_meta_2019_04_20.txt".
        area_gdf_file (str, optional): Selected bounding box's shape filename. Defaults to "output/selected_area_gdf.shp".
        nodes_file (str, optional): Real nodes' csv file. Defaults to "output/nodes_data.csv".
        links_df_file (str, optional): Real links' csv file. Defaults to "output/link_list.geojson".
        dist_threshold (int, optional): Maximum distance between traffic counter and link. Defaults to 30.
        counter_link_joined_data_file (str, optional): Filename for storing matched links with traffic counters. Defaults to "output/spatiotemporal_link_data.csv".
        dir (str, optional): Files directory. Defaults to "./examples/notebooks/data/".
    """

    def get_link_direction(row):
        u, v = row['u'], row['v']
        point1= nodes_gdf[nodes_gdf.osmid==u].geometry.iloc[0]
        lon1, lat1 = point1.x, point1.y
        point2= nodes_gdf[nodes_gdf.osmid==v].geometry.iloc[0]
        lon2, lat2 = point2.x, point2.y 
    
        direction_list = ["N", "E", "S", "W"]
        angle = atan2(cos(lat1)*sin(lat2)-sin(lat1) * cos(lat2)*cos(lon2-lon1), sin(lon2-lon1)*cos(lat2))
        bearing = (degrees(angle) + 360) % 360
        index = round(bearing/90)
        # Return direction_list[index%4]
        return [direction_list[index%4], direction_list[(index-1)%4], direction_list[(index+1)%4]]
    
    station_metadata = pd.read_csv(os.path.join(dir, station_metadata_file), sep='\t')
    
    station_md = gpd.GeoDataFrame(station_metadata, geometry=gpd.points_from_xy(station_metadata.Longitude, station_metadata.Latitude), 
                                  crs=network_crs)
    # Selected area
    area = gpd.read_file(os.path.join(dir, area_gdf_file))
    fresno_station = gpd.tools.sjoin(station_md, area, how='inner', predicate='within')
    
    # Nodes file
    nodes = pd.read_csv(os.path.join(dir, nodes_file))
    nodes_gdf = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.x, nodes.y))
    
    with open(os.path.join(dir, links_df_file), "r") as f:
        features = json.load(f)
        links = gpd.GeoDataFrame.from_features(features, crs=network_crs)
    
    links['direction'] = links.apply(get_link_direction, axis=1)
    
    nearest_link = gpd.sjoin_nearest(fresno_station.drop(['index_right'], axis=1).to_crs('EPSG:2228'), 
                                      links.to_crs("EPSG:2228"), rsuffix='links', lsuffix='station', distance_col='distance')
    nearest_link['distance'] = nearest_link['distance'] * 0.3048
    nearest_link.to_crs(network_crs, inplace=True)
    
    threshold = dist_threshold  # meters
    nearest_link = nearest_link[nearest_link['distance'] < threshold]
    # remove OR type and keep ML only
    nearest_link = nearest_link[nearest_link.Type == "ML"]
    
    group_links = nearest_link.groupby('index_links')
    groups_having_gt_1 = set(group_links.filter(lambda x: len(x) > 1)['index_links'])
    
    station_to_remove = []
    for st in groups_having_gt_1:
        group_df = group_links.get_group(st)
        group_df['select'] = group_df.apply(lambda row: True if row['Dir'] in row['direction'] else False, axis=1)
        station_to_remove.extend(list(group_df[group_df['select'] == False]['ID']))
    
    # Remove stations having wrong direction
    nearest_link = nearest_link[~nearest_link['ID'].isin(station_to_remove)]
    
    group_links = nearest_link.groupby('index_links')
    station_ids = list(group_links.first()['ID'])
    final_links = nearest_link[nearest_link['ID'].isin(station_ids)]
    
    print("Matched links with traffic counters:{0}".format(final_links.shape[0]))
    
    with open(os.path.join(dir, counter_link_joined_data_file), "w") as f:
        json.dump(final_links.__geo_interface__, f) 


def match_condition_check(row):
    dist, dist_5 = row['distance'], row['distance_5']
    u, v = row['u'], row['v']
    u_5, v_5 = row['u_5'], row['v_5']

    if str(dist_5) == 'nan':
        return True
    elif (u==u_5) and (v==v_5) and (dist <= dist_5):
        return True
    else: 
        return False


def traffic_counter_to_link_verify(counter_link_joined_data_file='output/station_links_joined_data.geojson',
                                  counter_link_joined_data_file_5='output/area_expansion/station_links_joined_data.geojson',
                                  dir="./examples/notebooks/data/"):   
    
    with open(os.path.join(dir, counter_link_joined_data_file_5), "r") as f:
        features = json.load(f)
        station_link_data_5 = gpd.GeoDataFrame.from_features(features, crs=network_crs)
        
    with open(os.path.join(dir, counter_link_joined_data_file), "r") as f:
        features = json.load(f)
        station_link_data = gpd.GeoDataFrame.from_features(features, crs=network_crs)
    	
    print("shape for matched traffic counter df:", station_link_data.shape, station_link_data_5.shape)
    
    station_df = pd.merge(left=station_link_data, right=station_link_data_5, how='left', on='ID', suffixes=('', '_5'))
    print("merged df shape:", station_df.shape)
    
    station_df = station_df[station_df.apply(lambda row: match_condition_check(row), axis=1)]
    
    station_link_data = station_link_data[station_link_data.ID.isin(list(station_df.ID))]
    print("matched traffic counters:", station_link_data.shape)
    
    with open(os.path.join(dir, counter_link_joined_data_file), "w") as f:
        json.dump(station_link_data.__geo_interface__, f) 
        

def traffic_counter_hour_data_by_link(counter_link_joined_data_file="output/station_links_joined_data.geojson",
                                      station_hour_data_file="input/d06_text_station_hour_2019_09.txt",
                                      spatiotemporal_link_data_file="output/spatiotemporal_link_data.csv",
                                      dir="./examples/notebooks/data/"):
    """Joins hourly traffic data with matched traffic counter links.

    Args:
        counter_link_joined_data_file (str, optional): matched links with traffic counters. Defaults to "output/station_links_joined_data.geojson".
        station_hour_data_file (str, optional): Pems hour data for traffic counters. Defaults to "input/d06_text_station_hour_2019_09.txt".
        spatiotemporal_link_data_file (str, optional): Filename for storing matched links with traffic counters. Defaults to "output/spatiotemporal_link_data.csv".
        dir (str, optional): Files directory. Defaults to "./examples/notebooks/data/".
    """
    with open(os.path.join(dir, counter_link_joined_data_file), "r") as f:
        features = json.load(f)
        station_link_data = gpd.GeoDataFrame.from_features(features, crs=network_crs)
    
    column_names = ["Timestamp", "Station", "District", "Route", "Direction", "Lane", "Station Length", "Samples", "% Observed", 
                    "Total Flow", "Avg Occupancy", "Avg Speed", "Delay (V_t=35)", "Delay (V_t=40)", "Delay (V_t=45)", "Delay (V_t=50)", 
                    "Delay (V_t=55)", "Delay (V_t=60)", "Lane 1 Flow Sum", "Lane 2 Flow Sum" , "Lane 3 Flow Sum", "Lane 4 Flow Sum", 
                    "Lane 5 Flow Sum", "Lane 6 Flow Sum", "Lane 7 Flow Sum", "Lane 8 Flow Sum", "Lane 1 Avg Occ", "Lane 2 Avg Occ", 
                    "Lane 3 Avg Occ", "Lane 4 Avg Occ", "Lane 5 Avg Occ", "Lane 6 Avg Occ", "Lane 7 Avg Occ", "Lane 8 Avg Occ", 
                    "Lane 1 Avg Speed", "Lane 2 Avg Speed", "Lane 3 Avg Speed", "Lane 4 Avg Speed", "Lane 5 Avg Speed", 
                    "Lane 6 Avg Speed", "Lane 7 Avg Speed", "Lane 8 Avg Speed"]
    
    station_hour_data = pd.read_csv(os.path.join(dir, station_hour_data_file), header=None, names=column_names)
    print("Shape of station hour data:{0}".format(station_hour_data.shape))

    df_m = pd.merge(station_link_data, station_hour_data, left_on=station_link_data.ID, right_on=station_hour_data.Station)
    
    df_m['link_key'] = df_m.apply(lambda row: (row['u'], row['v'], '0'), axis=1)
    required_columns = ['link_key', 'Timestamp', 'Station', 'Fwy', 'Dir', 'District_x', 'County', 'City',
           'State_PM', 'Abs_PM', 'Latitude', 'Longitude', 'Length', 'Type',
           'Lanes', 'Name', 'User_ID_1', 'User_ID_2', 'User_ID_3', 'User_ID_4',
           'city', 'osmid', 'u', 'v', 'length', 'highway',
           'maxspeed', 'lanes', 'capacity', 'direction', 'distance', 'Route', 'Direction', 'Lane', 'Station Length', 'Samples',
           '% Observed', 'Total Flow', 'Avg Occupancy', 'Avg Speed',
           'Delay (V_t=35)', 'Delay (V_t=40)', 'Delay (V_t=45)', 'Delay (V_t=50)',
           'Delay (V_t=55)', 'Delay (V_t=60)', 'Lane 1 Flow Sum',
           'Lane 2 Flow Sum', 'Lane 3 Flow Sum', 'Lane 4 Flow Sum',
           'Lane 5 Flow Sum', 'Lane 6 Flow Sum', 'Lane 7 Flow Sum',
           'Lane 8 Flow Sum', 'Lane 1 Avg Occ', 'Lane 2 Avg Occ', 'Lane 3 Avg Occ',
           'Lane 4 Avg Occ', 'Lane 5 Avg Occ', 'Lane 6 Avg Occ', 'Lane 7 Avg Occ',
           'Lane 8 Avg Occ', 'Lane 1 Avg Speed', 'Lane 2 Avg Speed',
           'Lane 3 Avg Speed', 'Lane 4 Avg Speed', 'Lane 5 Avg Speed',
           'Lane 6 Avg Speed', 'Lane 7 Avg Speed', 'Lane 8 Avg Speed']
    spatiotemporal_link_data = df_m[required_columns]
    print("Shape of spatiotemporal link data for selected links:{0}".format(spatiotemporal_link_data.shape))
    spatiotemporal_link_data.to_csv(os.path.join(dir, spatiotemporal_link_data_file), sep='\t', index=False, encoding='utf-8')


def generate_virtual_links_with_ct(census_dir="input/census_data", 
                                   census_shp_file="input/census_data/tl_2018_06_tract.shp",
                                   nodes_file="output/nodes_data.csv",
                                   links_df_file="output/link_list.geojson",
                                   max_vir_links=10,
                                   vir_link_cap=999999,
                                   state_id="06",
                                   year=2018,
                                   vir_links_file="output/virtual_links.csv", 
                                   vir_nodes_file="output/virtual_nodes.csv",
                                   url="https://www2.census.gov/geo/tiger/TIGER2018/TRACT/tl_2018_06_tract.zip",
                                   dir="./examples/notebooks/data/"):
    """Creates virtual nodes for each CT centers and generates n virtual links between CT center and real nodes.

    Args:
        census_dir (str, optional): Census data download directory. Defaults to "input/census_data".
        census_shp_file (str, optional): Census shape file. Defaults to "input/census_data/tl_2018_06_tract.shp".
        nodes_file (str, optional): Real nodes' csv file. Defaults to "output/nodes_data.csv".
        links_df_file (str, optional): Real links' shape file. Defaults to "output/link_list.geojson".
        max_vir_links (int, optional): Number of maximum virtual links per ct. Defaults to 10.
        state_id (str, optional): State id to download census data. Defaults to "06".
        year (int, optional): Year of which census data needs to be downloaded. Defaults to 2018.
        vir_links_file (str, optional): Filename to store generated virtual links to csv. Defaults to "output/virtual_links.csv".
        vir_nodes_file (str, optional): Filename to store generated virtual nodes to csv. Defaults to "output/virtual_nodes.csv".
        url (str, optional): URL to download CT zip file. Defaults to "https://www2.census.gov/geo/tiger/TIGER2018/TRACT/tl_2018_06_tract.zip".
        dir (str, optional): Files directory. Defaults to "./examples/notebooks/data/".
    """
    c = Census(_credential("CENSUS_API_KEY", API_KEY))
    census_data = c.acs5.state_county_tract(fields = ('NAME', 'B09019_001E'),
                                          state_fips = state_id,
                                          county_fips = "*",
                                          tract = "*",
                                          year = year)
    # Create a dataframe from the census data
    cdata_df = pd.DataFrame(census_data)
    cdata_df["GEOID"] = cdata_df["state"] + cdata_df["county"] + cdata_df["tract"]
       
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    # Check if directory exists if not create it
    if not os.path.exists(os.path.join(dir, census_dir)):
        print("creating directory at {0}".format(os.path.join(dir, census_dir)))
        os.mkdir(os.path.join(dir, census_dir))
    z.extractall(os.path.join(dir, census_dir))
    
    cinfo_df = gpd.read_file(os.path.join(dir + census_shp_file))
    cinfo_df = cinfo_df.to_crs(network_crs)
    
    ctract_merge = cinfo_df.merge(cdata_df, on = "GEOID")
    ctract_merge.shape
    ctract_merge['g_ct'] = ctract_merge['geometry']
    fresno_county = ctract_merge.copy(deep=True)
    
    # Read the nodes and links data provided which are already in the bounding box
    nodes = pd.read_csv(os.path.join(dir, nodes_file))
    
    with open(os.path.join(dir, links_df_file), "r") as f:
        features = json.load(f)
        links = gpd.GeoDataFrame.from_features(features, crs=network_crs)
    
    # Create the geometry for the nodes 
    nodes = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.x, nodes.y), crs=network_crs)
    gdf_selected_ct = gpd.sjoin(gpd.GeoDataFrame(nodes), gpd.GeoDataFrame(fresno_county) , how="inner", predicate='within')
    gdf_selected_ct['centroid'] = gpd.GeoSeries(gdf_selected_ct['g_ct'], crs=network_crs).centroid
    
    gdf_selected_taz = gdf_selected_ct.copy(deep=True)
    
    # Virtual links creation
    new_links = gdf_selected_taz.copy(deep=True)
    new_links['line'] = new_links.apply(lambda row: LineString([row['geometry'], row['centroid']]), axis=1)
    
    new_links['length'] = gpd.GeoSeries(new_links['line']).set_crs(network_crs).to_crs('EPSG:2228').length * 0.3048
    new_links.sort_values(by=['length'], inplace=True)
    new_links = new_links.groupby('GEOID').nth[:max_vir_links]
    print("Generated links shape:{0} for {1} cts".format(new_links.shape, gdf_selected_ct.GEOID.nunique()))
    
    # Virtual nodes
    unique_centroid = new_links[~new_links.GEOID.duplicated(keep='first')][['centroid', 'tract', 'g_ct']]
    # unique_centroid = gpd.GeoDataFrame(unique_centroid, geometry=unique_centroid.centroid, crs=network_crs)
    unique_centroid['lon'] = unique_centroid.centroid.apply(lambda p:p.x)
    unique_centroid['lat'] = unique_centroid.centroid.apply(lambda p:p.y)
    unique_centroid['type'] = 'attractor'
    print("Generated nodes shape:{0} for {1} cts".format(unique_centroid.shape, gdf_selected_ct.GEOID.nunique()))
    
    # Generate the list of link osmid's to get the maximum value
    links_osmid_list = []
    for i in links.osmid:
        if type(i) == type([]):
            links_osmid_list.extend(i)
        else:
            links_osmid_list.append(i)
    
    # Generate the osmid for the virtual nodes that start from max_osmid (ie max between osmid in nodes and links)
    max_osmid =  max(list(nodes.osmid) + links_osmid_list) +  1
    unique_centroid['osmid'] = range(max_osmid, max_osmid + unique_centroid.shape[0]) 
    
    # attractor_virtual_nodes = unique_centroid[['osmid', 'type', 'lon', 'lat', 'centroid']].rename(columns={'centroid': 'geometry'})
    attractor_virtual_nodes = gpd.GeoDataFrame(unique_centroid[['osmid', 'type', 'lon', 'lat']], 
                                           geometry=unique_centroid.centroid, crs=network_crs)
    
    # Here osmid is renamed to u because its came from real node 
    attractor_virtual_links = new_links[['osmid', 'centroid', 'length']].rename(columns={'osmid': 'u'})
    attractor_virtual_links = gpd.GeoDataFrame(attractor_virtual_links, geometry=attractor_virtual_links.centroid, crs=network_crs)

    real_to_vir_nodes_map = pd.merge(attractor_virtual_links, attractor_virtual_nodes, on='geometry')[['u', 'osmid']].set_index('u').to_dict()
    
    # attractor_virtual_links['v'] = list(pd.merge(attractor_virtual_links, attractor_virtual_nodes, on='geometry')['osmid'])
    attractor_virtual_links['v'] = attractor_virtual_links.u.apply(lambda val: real_to_vir_nodes_map['osmid'][val])
    attractor_virtual_links['link_key'] = attractor_virtual_links.apply(lambda x: (x['u'], x['v'], '0'), axis=1)
    attractor_virtual_links['link_type'] = 'attractor'
    
    generator_virtual_links = attractor_virtual_links.copy(deep=True)
    generator_virtual_links.rename(columns={'u': 'v', 'v': 'u'}, inplace=True)
    generator_virtual_links['link_type'] = 'generator'
    generator_virtual_links['link_key'] = generator_virtual_links.apply(lambda x: (x['u'], x['v'], '0'), axis=1)
    generator_virtual_links = generator_virtual_links[attractor_virtual_links.columns]
    generator_virtual_links['u'] = generator_virtual_links.u.apply(lambda a: a + unique_centroid.shape[0])
    
    generator_virtual_nodes = generator_virtual_links[~generator_virtual_links.u.duplicated(keep='first')]
    generator_virtual_nodes = generator_virtual_nodes[['u', 'geometry']].rename(columns={'u': 'osmid'})
    generator_virtual_nodes['lon'] = generator_virtual_nodes.geometry.x
    generator_virtual_nodes['lat'] = generator_virtual_nodes.geometry.y
    generator_virtual_nodes['type'] = 'generator'
    generator_virtual_nodes = generator_virtual_nodes[attractor_virtual_nodes.columns]
    
    virtual_links = pd.concat([attractor_virtual_links, generator_virtual_links], ignore_index=True)
    max_links_osmid = max(generator_virtual_nodes.osmid)
    virtual_links['osmid'] = range(max_links_osmid, max_links_osmid + virtual_links.shape[0]) 
    
    # Set the values for lanes to inf and maxspeed to inf for the virtual links
    virtual_links['lanes'] = np.inf
    virtual_links['maxspeed'] =  np.inf
    virtual_links['capacity'] =  vir_link_cap
    virtual_links = virtual_links[['osmid', 'link_key', 'length', 'u', 'v', 'link_type', 'lanes', 'capacity', 'maxspeed']]
    print("Virtual links shape:{0}".format(virtual_links.shape))
    virtual_links.to_csv(os.path.join(dir, vir_links_file), index=False)

    virtual_nodes = pd.concat([attractor_virtual_nodes, generator_virtual_nodes], ignore_index=True)
    print("Virtual nodes shape:{0}".format(virtual_nodes.shape))
    
    # Save the virtual nodes file 
    virtual_nodes.to_csv(os.path.join(dir, vir_nodes_file), index=False)


def generate_virtual_links_with_taz(taz_shape_file="input/TAZ_Shap_Files/tz06_d00_shp/tz06_d00.shp",
                                    nodes_file="output/nodes_data.csv",
                                    links_df_file="output/link_list.geojson",
                                    max_vir_links=10,
                                    vir_link_cap=999999,
                                    vir_links_file="output/virtual_links.csv", 
                                    vir_nodes_file="output/virtual_nodes.csv",
                                    dir="./examples/notebooks/data/"):
    """Creates virtual nodes for each TAZ centers and generates n virtual links between TAZ center and real nodes.

    Args:
        taz_shape_file (str, optional): Traffic Analysis Zone file for US. Defaults to "input/TAZ_Shap_Files/tz06_d00_shp/tz06_d00.shp".
        nodes_file (str, optional): Real nodes' csv file. Defaults to "output/nodes_data.csv".
        links_df_file (str, optional): Real links' shape file. Defaults to "output/link_list.geojson".
        max_vir_links (int, optional): Number of maximum virtual links per taz. Defaults to 10.
        vir_links_file (str, optional): Filename to store generated virtual links to csv. Defaults to "output/virtual_links.csv".
        vir_nodes_file (str, optional): Filename to store generated virtual nodes to csv. Defaults to "output/virtual_nodes.csv".
        dir (str, optional): Files directory. Defaults to "./examples/notebooks/data/".
    """
    
    # Read the TAZ input files
    gdf_taz = gpd.read_file(os.path.join(dir, taz_shape_file))
    gdf_taz = gdf_taz.set_crs(network_crs)
    
    # Read the nodes and links data provided which are already in the bounding box
    nodes = pd.read_csv(os.path.join(dir, nodes_file))
    
    with open(os.path.join(dir, links_df_file), "r") as f:
        features = json.load(f)
        links = gpd.GeoDataFrame.from_features(features, crs=network_crs)
    
    # Create the geometry for the nodes 
    nodes = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.x, nodes.y), crs=network_crs)
    
    gdf_taz['geometry_polygon'] = gdf_taz.geometry
    gdf_selected_taz = gpd.sjoin(gpd.GeoDataFrame(nodes), gpd.GeoDataFrame(gdf_taz) , how="inner", predicate='within')
    
    # Now calculate the centroids
    gdf_selected_taz['centroid'] = gpd.GeoSeries(gdf_selected_taz['geometry_polygon'], crs=network_crs).centroid
    
    # Virtual links
    new_links = gdf_selected_taz.copy(deep=True)
    new_links['line'] = new_links.apply(lambda row: LineString([row['geometry'], row['centroid']]), axis=1)
    
    new_links['length'] = gpd.GeoSeries(new_links['line']).set_crs(network_crs).to_crs('EPSG:2228').length * 0.3048
    new_links.sort_values(by=['length'], inplace=True)
    new_links = new_links.groupby('TAZ').nth[:max_vir_links]
    print("Generated links shape:{0} for {1} tazs".format(new_links.shape, gdf_selected_taz.TAZ.nunique()))
    
    # Virtual nodes from links
    unique_centroid = new_links[~new_links.TAZ.duplicated(keep='first')][['centroid', 'TAZ', 'geometry_polygon']]
    unique_centroid['lon'] = unique_centroid.centroid.apply(lambda p:p.x)
    unique_centroid['lat'] = unique_centroid.centroid.apply(lambda p:p.y)
    unique_centroid['type'] = 'attractor'
    print("Generated centroids shape:{0} for {1} tazs".format(unique_centroid.shape, gdf_selected_taz.TAZ.nunique()))
    
    # Generate the list of link osmid's to get the maximum value
    links_osmid_list = []
    for i in links.osmid:
        if type(i) == type([]):
            links_osmid_list.extend(i)
        else:
            links_osmid_list.append(i)
    
    # Generate the osmid for the virtual nodes that start from max_osmid (ie max between osmid in nodes and links)
    max_osmid =  max(list(nodes.osmid) + links_osmid_list) +  1
    unique_centroid['osmid'] = range(max_osmid, max_osmid + unique_centroid.shape[0]) 
    # attractor_virtual_nodes = unique_centroid[['osmid', 'type', 'lon', 'lat', 'centroid']].rename(columns={'centroid': 'geometry'})
    attractor_virtual_nodes = gpd.GeoDataFrame(unique_centroid[['osmid', 'type', 'lon', 'lat']], 
                                           geometry=unique_centroid.centroid, crs=network_crs)
    
    # Here osmid is renamed to u because its came from real node 
    attractor_virtual_links = new_links[['osmid', 'centroid', 'length']].rename(columns={'osmid': 'u'})
    attractor_virtual_links = gpd.GeoDataFrame(attractor_virtual_links, geometry=attractor_virtual_links.centroid, crs=network_crs)
    
    real_to_vir_nodes_map = pd.merge(attractor_virtual_links, attractor_virtual_nodes, on='geometry')[['u', 'osmid']].set_index('u').to_dict()
    
    # attractor_virtual_links['v'] = list(pd.merge(attractor_virtual_links, attractor_virtual_nodes, on='geometry')['osmid'])
    attractor_virtual_links['v'] = attractor_virtual_links.u.apply(lambda val: real_to_vir_nodes_map['osmid'][val])
    attractor_virtual_links['link_key'] = attractor_virtual_links.apply(lambda x: (x['u'], x['v'], '0'), axis=1)
    attractor_virtual_links['link_type'] = 'attractor'
    attractor_virtual_links.shape
    
    generator_virtual_links = attractor_virtual_links.copy(deep=True)
    generator_virtual_links.rename(columns={'u': 'v', 'v': 'u'}, inplace=True)
    generator_virtual_links['link_type'] = 'generator'
    generator_virtual_links['link_key'] = generator_virtual_links.apply(lambda x: (x['u'], x['v'], '0'), axis=1)
    generator_virtual_links = generator_virtual_links[attractor_virtual_links.columns]
    
    generator_virtual_links['u'] = generator_virtual_links.u.apply(lambda a: a + unique_centroid.shape[0])
    
    generator_virtual_nodes = generator_virtual_links[~generator_virtual_links.u.duplicated(keep='first')]
    generator_virtual_nodes = generator_virtual_nodes[['u', 'geometry']].rename(columns={'u': 'osmid'})
    generator_virtual_nodes['lon'] = generator_virtual_nodes.geometry.x
    generator_virtual_nodes['lat'] = generator_virtual_nodes.geometry.y
    generator_virtual_nodes['type'] = 'generator'
    generator_virtual_nodes = generator_virtual_nodes[attractor_virtual_nodes.columns]
    
    virtual_links = pd.concat([attractor_virtual_links, generator_virtual_links], ignore_index=True)
    print("Virtual links shape:{0}".format(virtual_links.shape))
    
    max_links_osmid = max(generator_virtual_nodes.osmid)
    virtual_links['osmid'] = range(max_links_osmid, max_links_osmid + virtual_links.shape[0]) 
    # Set the values for lanes to inf and maxspeed to inf for the virtual links
    virtual_links['lanes'] = np.inf
    virtual_links['maxspeed'] =  np.inf
    virtual_links['capacity'] =  vir_link_cap
    virtual_links = virtual_links[['osmid', 'link_key', 'length', 'u', 'v', 'link_type', 'lanes', 'capacity', 'maxspeed']]
    
    virtual_links.to_csv(os.path.join(dir, vir_links_file), index=False)
    
    virtual_nodes = pd.concat([attractor_virtual_nodes, generator_virtual_nodes], ignore_index=True)
    print("Virtual nodes shape:{0}".format(virtual_nodes.shape))
    # Save the virtual nodes to file 
    virtual_nodes.to_csv(os.path.join(dir, vir_nodes_file), index=False)


def combine_nodes_n_links(nodes_file="output/nodes_data.csv", 
                          links_file="output/links_data.csv", 
                          vir_links_file="output/virtual_links.csv", 
                          vir_nodes_file="output/virtual_nodes.csv",
                          node_specific_data_file="output/node_specific_data.csv",
                          link_specific_data_file="output/link_specific_data.csv",
                          dir="./examples/notebooks/data/"):
    """Combines all virtual and real nodes and links to csv files.

    Args:
        nodes_file (str, optional): Real nodes' csv file. Defaults to "output/nodes_data.csv".
        links_file (str, optional): Real links' csv file. Defaults to "output/links_data.csv".
        vir_links_file (str, optional): Virtual links' csv file. Defaults to "output/virtual_links.csv".
        vir_nodes_file (str, optional): Virtual nodes' csv file. Defaults to "output/virtual_nodes.csv".
        node_specific_data_file (str, optional): Filename to store all nodes data to csv. Defaults to "output/node_specific_data.csv".
        link_specific_data_file (str, optional): Filename to store all links data to csv. Defaults to "output/link_specific_data.csv".
        dir (str, optional): Files directory. Defaults to "./examples/notebooks/data/".
    """    
    # Real nodes
    real_nodes = pd.read_csv(os.path.join(dir, nodes_file))
    real_nodes['type'] = 'real'
    real_nodes.rename(columns={"x": "lon", "y": "lat"}, inplace=True)
    
    vir_nodes = pd.read_csv(os.path.join(dir, vir_nodes_file))
    vir_nodes = vir_nodes[real_nodes.columns]
    print("Real nodes:{0} and Virtual nodes:{1} in network".format(real_nodes.shape, vir_nodes.shape))
    
    nodes = pd.concat([real_nodes, vir_nodes])
    nodes.reset_index(inplace=True)
    nodes['key'] = list(range(nodes.shape[0]))
    nodes = nodes[['key', 'osmid', 'type', 'lon', 'lat']]
    print("Combined nodes:{0}".format(nodes.shape))
    nodes.to_csv(os.path.join(dir, node_specific_data_file), encoding='utf-8', sep='\t', index=False)
    
    # Links data
    real_links = pd.read_csv(os.path.join(dir, links_file))
    real_links["link_type"] = "street"
    vir_links = pd.read_csv(os.path.join(dir, vir_links_file))
    vir_links['highway'] = np.nan
    vir_links = vir_links[real_links.columns]
    print("Real links:{} and Virtual links:{} in network".format(real_links.shape, vir_links.shape))
    
    links = pd.concat([real_links, vir_links])
    links.sort_values(by=['u', 'v'], inplace=True)
    links_updated = pd.merge(links, nodes, left_on='u', right_on='osmid').rename(columns={"osmid_x": "osmid_link", 
                                                                                          'key': 'key_u', 'lon': 'lon_u', 'lat': 'lat_u'})
    links_updated.drop(columns=['osmid_y', 'type'], inplace=True)
    links_updated_final = pd.merge(links_updated, nodes, left_on='v', 
                                   right_on='osmid').rename(columns={'key': 'key_v', 'lon': 'lon_v', 
                                                                     'lat': 'lat_v'}).sort_values(by=['key_u', 'key_v'])
    links_updated_final.drop(columns=['osmid', 'type'], inplace=True)
    
    links_updated_final['link_key'] = links_updated_final.apply(lambda row: (row['key_u'], row['key_v'], '0'), axis=1)
    links_updated_final['link_keys_osmids'] = links_updated_final.apply(lambda row: (row['u'], row['v'], '0'), axis=1)
    
    links_updated_final = links_updated_final[['link_key', 'link_keys_osmids', 'osmid_link', 'u', 'v', 'length', 'maxspeed', 
                                               'lanes', 'capacity', 'link_type', 'key_u', 'key_v', 'lon_u', 'lat_u', 'lon_v', 'lat_v']]
    # Define a function to extract the number from a string
    def extract_number(s):
        try:
            numbers = re.findall(r'[0-9\.]+', s)
            return float(numbers[0]) if numbers else None
        except:
            pass #print(s)
    # Apply the function to all non-null values in 'maxspeed'
    links_updated_final.loc[links_updated_final['maxspeed'].notnull(), 'maxspeed_float'] = links_updated_final.loc[links_updated_final['maxspeed'].notnull(), 'maxspeed'].apply(extract_number)
    
    # Create a SimpleImputer object with strategy as 'mean'
    imputer = SimpleImputer(strategy='mean')
    
    # Select the column 'maxspeed'
    maxspeed = links_updated_final[['maxspeed_float']]
    
    # Perform mean imputation on 'maxspeed'
    maxspeed_imputed = imputer.fit_transform(maxspeed)
    
    # Keep imputed values in another field
    links_updated_final['maxspeed_impute'] = maxspeed_imputed
    
    if 'tf_inrix' not in links_updated_final.columns:
        links_updated_final['tf_inrix'] = (60 * 60 / 1000) * links_updated_final['length'].astype('float')/(links_updated_final['maxspeed_impute'].astype('float') * 1.60934)
    
    # links_updated_final['capacity'] = links_updated_final.lanes.apply(extract_number).apply(lambda x: 2000 if pd.isna(x) else x*2000)
    
    print("Combined links:{0}".format(links_updated_final.shape))
    links_updated_final.to_csv(os.path.join(dir, link_specific_data_file), encoding='utf-8', sep='\t', index=False)


def link_specific_null_values(spatiotemporal_link_data_file="output/spatiotemporal_link_data.csv", 
                              link_specific_data_file="output/link_specific_data.csv",
                              spatiotemporal_link_data_whole_file="output/spatiotemporal_link_data_whole.csv",
                              dir="./examples/notebooks/data/"):
    """Creates final links data file to train model with hourly data and adds null values where no station counter present.

    Args:
        spatiotemporal_link_data_file (str, optional): Matched links file with traffic counters. Defaults to "output/spatiotemporal_link_data.csv".
        link_specific_data_file (str, optional): Network links data csv file. Defaults to "output/link_specific_data.csv".
        spatiotemporal_link_data_whole_file (str, optional): Filename to store whole link specific hourly data to csv. Defaults to "output/spatiotemporal_link_data_whole.csv".
        dir (str, optional): Files directory. Defaults to "./examples/notebooks/data/".
    """
    spatiotemporal_link_data = pd.read_csv(os.path.join(dir, spatiotemporal_link_data_file), sep='\t')
    spatiotemporal_link_data.rename(columns={'link_key': 'link_keys_osmids', 'osmid': 'osmid_link'}, inplace=True)
    spatiotemporal_link_data['link_type'] = 'street'
    
    link_specific_data = pd.read_csv(os.path.join(dir, link_specific_data_file), sep='\t')
    
    spatiotemporal_data_1 = pd.merge(spatiotemporal_link_data, link_specific_data[['link_key', 'link_keys_osmids', 'key_u', 'key_v', 
                            'lon_u', 'lat_u', 'lon_v', 'lat_v', 'maxspeed_float', 
                            'maxspeed_impute', 'tf_inrix', 'capacity']], on='link_keys_osmids').sort_values(by=['key_u', 'key_v', 'Timestamp'])
    
    spatiotemporal_data_1 = spatiotemporal_data_1[['link_key', 'link_keys_osmids', 'Timestamp', 'Station', 'Fwy', 'Dir', 'District_x',
           'County', 'City', 'State_PM', 'Abs_PM', 'Latitude', 'Longitude', 'Length', 'Type', 'Lanes', 'Name', 'User_ID_1', 'User_ID_2',
           'User_ID_3', 'User_ID_4', 'city', 'osmid_link', 'u', 'v', 'length', 'highway', 'maxspeed', 'lanes', 'link_type', 'key_u', 'key_v',
            'lon_u', 'lat_u', 'lon_v', 'lat_v', 'maxspeed_float', 'maxspeed_impute', 'tf_inrix', 'capacity_y', 'direction', 'distance', 'Route',
           'Direction', 'Lane', 'Station Length', 'Samples', '% Observed', 'Total Flow', 'Avg Occupancy', 'Avg Speed', 'Delay (V_t=35)',
           'Delay (V_t=40)', 'Delay (V_t=45)', 'Delay (V_t=50)', 'Delay (V_t=55)', 'Delay (V_t=60)', 'Lane 1 Flow Sum', 'Lane 2 Flow Sum',
           'Lane 3 Flow Sum', 'Lane 4 Flow Sum', 'Lane 5 Flow Sum', 'Lane 6 Flow Sum', 'Lane 7 Flow Sum', 'Lane 8 Flow Sum',
           'Lane 1 Avg Occ', 'Lane 2 Avg Occ', 'Lane 3 Avg Occ', 'Lane 4 Avg Occ', 'Lane 5 Avg Occ', 'Lane 6 Avg Occ', 'Lane 7 Avg Occ', 
           'Lane 8 Avg Occ', 'Lane 1 Avg Speed', 'Lane 2 Avg Speed', 'Lane 3 Avg Speed',
           'Lane 4 Avg Speed', 'Lane 5 Avg Speed', 'Lane 6 Avg Speed', 'Lane 7 Avg Speed', 'Lane 8 Avg Speed']]
    
    spatiotemporal_data_1.rename(columns={'capacity_y':'capacity'}, inplace=True)
    
    present_links = set(list(spatiotemporal_data_1.link_keys_osmids))
    print("Matched links with traffic counters:{0}".format(len(set(list(spatiotemporal_data_1.link_keys_osmids)))))
    timestamps = pd.DataFrame(spatiotemporal_link_data.Timestamp.value_counts()).reset_index()
    timestamps['count'] = 1
    remaining_df = link_specific_data[link_specific_data.apply(lambda row: False if row['link_keys_osmids'] in present_links else True, axis=1)]
    print("Remaining links:{0}".format(remaining_df.shape[0]))

    remaining_df['count'] = 1
    remaining_link_data = pd.merge(remaining_df, timestamps, on='count')
    remaining_link_data.sort_values(by=["u", 'v', 'Timestamp'], inplace=True)
    blank_columns = list(i for i in spatiotemporal_data_1.columns if i not in remaining_link_data.columns)
    remaining_link_data[blank_columns] = np.nan
    remaining_link_data = remaining_link_data[['link_key', 'link_keys_osmids', 'Timestamp', 'Station', 'Fwy', 'Dir', 'District_x',
           'County', 'City', 'State_PM', 'Abs_PM', 'Latitude', 'Longitude', 'Length', 'Type', 'Lanes', 'Name', 'User_ID_1', 'User_ID_2',
           'User_ID_3', 'User_ID_4', 'city', 'osmid_link', 'u', 'v', 'length', 'highway', 'maxspeed', 'lanes', 'link_type', 'key_u', 
           'key_v', 'lon_u', 'lat_u', 'lon_v', 'lat_v', 'maxspeed_float', 'maxspeed_impute', 'tf_inrix', 'capacity', 'direction', 'distance', 'Route',
           'Direction', 'Lane', 'Station Length', 'Samples', '% Observed', 'Total Flow', 'Avg Occupancy', 'Avg Speed', 'Delay (V_t=35)',
           'Delay (V_t=40)', 'Delay (V_t=45)', 'Delay (V_t=50)', 'Delay (V_t=55)', 'Delay (V_t=60)', 'Lane 1 Flow Sum', 'Lane 2 Flow Sum',
           'Lane 3 Flow Sum', 'Lane 4 Flow Sum', 'Lane 5 Flow Sum', 'Lane 6 Flow Sum', 'Lane 7 Flow Sum', 'Lane 8 Flow Sum',
           'Lane 1 Avg Occ', 'Lane 2 Avg Occ', 'Lane 3 Avg Occ', 'Lane 4 Avg Occ', 'Lane 5 Avg Occ', 'Lane 6 Avg Occ', 'Lane 7 Avg Occ', 
           'Lane 8 Avg Occ', 'Lane 1 Avg Speed', 'Lane 2 Avg Speed', 'Lane 3 Avg Speed',
           'Lane 4 Avg Speed', 'Lane 5 Avg Speed', 'Lane 6 Avg Speed', 'Lane 7 Avg Speed', 'Lane 8 Avg Speed']]
    spatiotemporal_final_data = pd.concat([spatiotemporal_data_1, remaining_link_data])
    print("Whole spatiotemporal link data: {0}".format(spatiotemporal_final_data.shape))
    spatiotemporal_final_data.sort_values(by=['key_u', 'key_v', 'Timestamp'], inplace=True)
    
    # To write the whole spatiotemporal data un-comment line below
    spatiotemporal_final_data.to_csv(os.path.join(dir, spatiotemporal_link_data_whole_file), 
                                     sep='\t', index=False, encoding='utf-8')


def prior_data_reference_q_taz(county = "Fresno, CA",
                               taz_shp_file="input/TAZ_Shap_Files/tz06_d00_shp/tz06_d00.shp",
                               node_specific_data_file="output/node_specific_data.csv",
                               vir_nodes_file="output/virtual_nodes.csv",
                               year=2018,
                               ref_g_file="output/ref_g.csv",
                               counties_shp=None,
                               dir="./examples/notebooks/data/"):
    """Generates ref_g metrics from average house hold trips in each taz.

    Args:
        county (str, optional) : County to get the household data. Defaults to "Fresno, CA"s
        taz_shp_file (str, optional): Traffic Analysis Zone file for US. Defaults to "input/TAZ_Shap_Files/tz06_d00_shp/tz06_d00.shp".
        node_specific_data_file (str, optional): Network nodes data csv file. Defaults to "output/node_specific_data.csv".
        vir_nodes_file (str, optional): Virtual nodes' csv file. Defaults to "output/virtual_nodes.csv".
        year (int, optional): Year of which census data for county needs to be downloaded. Defaults to 2018.
        ref_g_file (str, optional): Reference_g metrics to generate prior trips data. Defaults to "output/ref_g.csv".
        dir (str, optional): Files directory. Defaults to "./examples/notebooks/data/".
    """
    if counties_shp is None:
        raise ValueError("counties_shp is required; 01_build_network.ipynb builds it.")
    fresno_county = gpd.read_file(os.path.join(dir, counties_shp))
    fresno_county['g_ct'] = fresno_county.geometry
    fresno_county['ct_area'] = fresno_county.geometry.area
    fresno_county['ct_den_mil'] = fresno_county.total_hh/(fresno_county.ct_area*10**6)
    
    # TAZ
    gdf = gpd.read_file(os.path.join(dir, taz_shp_file))
    gdf = gdf.set_crs(network_crs)
    
    nodes = pd.read_csv(os.path.join(dir, node_specific_data_file), sep='\t')
    nodes_gdf = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.lon, nodes.lat, crs=network_crs))
    
    virtual_nodes_df = pd.read_csv(os.path.join(dir, vir_nodes_file))
    
    gdf['unique_key'] = list(range(gdf.shape[0]))
        
    fresno_taz = gpd.sjoin(nodes_gdf[['key', 'geometry']], gdf, how='inner', predicate='within')
    selected_taz = gdf[gdf.unique_key.isin(fresno_taz.unique_key.unique())]
    selected_taz['taz_area'] = selected_taz.geometry.area
    selected_taz['g_taz'] = selected_taz.geometry
    
    fresno_taz_ct = gpd.tools.sjoin(selected_taz, fresno_county, how='inner', predicate='intersects', lsuffix='_taz', rsuffix='_ct')
    #print(fresno_taz_ct.shape)
    fresno_taz_ct['per'] = fresno_taz_ct.apply(lambda row: row.g_taz.intersection(row.g_ct).area/row.g_taz.area, axis=1) 
    fresno_taz_ct['hh_by_ct'] = fresno_taz_ct.taz_area * fresno_taz_ct.ct_den_mil * 10**6
    fresno_taz_ct['hh_by_per'] = fresno_taz_ct.hh_by_ct * fresno_taz_ct.per
    
    group_taz = fresno_taz_ct.groupby(by=['TAZ']).agg({'hh_by_per':sum})
    city_taz_with_hh = pd.merge(selected_taz, group_taz, how='inner', on='TAZ')
    
    virtual_nodes_df = virtual_nodes_df[virtual_nodes_df.type == 'generator']
    
    # Create the geometry for the virtual nodes 
    virtual_nodes = gpd.GeoDataFrame(virtual_nodes_df, geometry=gpd.points_from_xy(virtual_nodes_df.lon, virtual_nodes_df.lat), 
                                     crs=network_crs)
    all_virtual_nodes  = gpd.sjoin(virtual_nodes, city_taz_with_hh, how="inner", predicate='within')
    print("Joined vir nodes:{0} with house hold data".format(all_virtual_nodes.shape))
    
    nodes_with_population = pd.merge(nodes,all_virtual_nodes[['osmid','hh_by_per']], how='left', on='osmid').sort_values(by=['key'])
    nodes_with_population.loc[nodes_with_population['hh_by_per'].isna(),'hh_by_per']=0
    nodes_with_population = nodes_with_population.rename(columns={'hh_by_per':'house_hold'})

    nodes_with_population[['key','osmid','house_hold']].to_csv(os.path.join(dir, ref_g_file), index=False)
 

def prior_data_reference_q_ct(census_dir="input/census_data", 
                              census_shp_file="input/census_data/tl_2018_06_tract.shp",
                              node_specific_data_file="output/node_specific_data.csv", 
                              vir_nodes_file="output/virtual_nodes.csv",
                              state_id="06",
                              year=2018,
                              ref_g_file="output/ref_g.csv", 
                              url="https://www2.census.gov/geo/tiger/TIGER2018/TRACT/tl_2018_06_tract.zip",
                              dir="./examples/notebooks/data/"):
    """Generates ref_g metrics from average house hold trips in each ct.

    Args:
        census_dir (str, optional): Census data download directory. Defaults to "input/census_data".
        census_shp_file (str, optional): Census shape file. Defaults to "input/census_data/tl_2018_06_tract.shp".
        node_specific_data_file (str, optional): Network nodes data csv file. Defaults to "output/node_specific_data.csv".
        vir_nodes_file (str, optional): Virtual nodes' csv file. Defaults to "output/virtual_nodes.csv".
        state_id (str, optional): State id to download census data. Defaults to "06".
        year (int, optional): Year of which census data needs to be downloaded. Defaults to 2018.
        ref_g_file (str, optional): Reference_g metrics to generate prior trips data. Defaults to "output/ref_g.csv".
        url (str, optional): URL to download CT zip file. Defaults to "https://www2.census.gov/geo/tiger/TIGER2018/TRACT/tl_2018_06_tract.zip".
        dir (str, optional): Files directory. Defaults to "./examples/notebooks/data/".
    """
    c = Census(_credential("CENSUS_API_KEY", API_KEY))
    census_data = c.acs5.state_county_tract(fields = ('NAME', 'B09019_001E'),
                                          state_fips = state_id,
                                          county_fips = "*",
                                          tract = "*",
                                          year = year)
    # Create a dataframe from the census data
    cdata_df = pd.DataFrame(census_data)
    cdata_df.shape
    cdata_df["GEOID"] = cdata_df["state"] + cdata_df["county"] + cdata_df["tract"]
    
    # If census zip is not downloaded in ct then only download run below code
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    # Check if directory exists if not create it
    if not os.path.exists(os.path.join(dir, census_dir)):
        print("creating directory at {0}".format(os.path.join(dir, census_dir)))
        os.mkdir(os.path.join(dir, census_dir))
    z.extractall(os.path.join(dir, census_dir))
    
    cinfo_df = gpd.read_file(os.path.join(dir, census_shp_file))
    nodes = pd.read_csv(os.path.join(dir, node_specific_data_file), sep='\t')
    virtual_nodes_df = pd.read_csv(os.path.join(dir, vir_nodes_file))
    
    nodes_gdf = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.lon, nodes.lat, crs=network_crs))
    
    cinfo_df = cinfo_df.to_crs(network_crs)
    ctract_merge = cinfo_df.merge(cdata_df, on = "GEOID")
    ctract_merge['g_ct'] = ctract_merge['geometry']
    
    fresno_county = ctract_merge.copy(deep=True)
    fresno_ct = gpd.sjoin(nodes_gdf[['key', 'geometry']], fresno_county, how='inner', predicate='within')
    selected_ct = fresno_county[fresno_county.GEOID.isin(fresno_ct.GEOID.unique())]
    print("House hold data shape:{0} for selected bounding box".format(selected_ct.shape))
    
    virtual_nodes_df = virtual_nodes_df[virtual_nodes_df.type == 'generator']
    # Create the geometry for the virtual nodes 
    virtual_nodes = gpd.GeoDataFrame(virtual_nodes_df, geometry=gpd.points_from_xy(virtual_nodes_df.lon, virtual_nodes_df.lat), crs=network_crs)
    
    all_virtual_nodes  = gpd.sjoin(virtual_nodes, selected_ct, how="inner", predicate='within')
    print("Joined vir nodes:{0} with house hold data".format(all_virtual_nodes.shape))
    
    nodes_with_population = pd.merge(nodes,all_virtual_nodes[['osmid','B09019_001E']], how='left', on='osmid').sort_values(by=['key'])
    nodes_with_population.rename(columns={'B09019_001E':'hh_by_per'}, inplace=True)
    nodes_with_population.loc[nodes_with_population['hh_by_per'].isna(),'hh_by_per']=0
    nodes_with_population = nodes_with_population.rename(columns={'hh_by_per':'house_hold'} )
    
    nodes_with_population[['key','osmid','house_hold']].to_csv(os.path.join(dir, ref_g_file), index=False)
