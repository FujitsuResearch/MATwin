# Module containing all the functions of folium map visualization & adding mutiple layers.

import folium
from folium import FeatureGroup, LayerControl
from folium.plugins import FeatureGroupSubGroup
from shapely import geometry
from shapely.geometry import LineString
from shapely.geometry import Point
from branca.colormap import LinearColormap
import branca.colormap as cm
import geopandas as gpd
import numpy as np
import json
import ast
import math
from collections import Counter
import networkx as nx

class BoundingBox(object):
    def __init__(self, *args, **kwargs):
        self.lat_min = None
        self.lon_min = None
        self.lat_max = None
        self.lon_max = None


def compute_alternate_path_without_links(links_df, origin, destination, forbidden_links,
                                         weight_col="bf_tt", forbid_reverse=True):
    """Return shortest path link keys avoiding the forbidden links.

    Parameters
    ----------
    links_df : geopandas.GeoDataFrame
        Link records with geometry and weight columns.
    origin, destination : int
        Node ids for the OD pair.
    forbidden_links : Iterable[tuple]
        Collection of link keys (u, v, key) to exclude from the search.
    weight_col : str, optional
        Column providing edge weights; falls back to length if unavailable.

    Returns
    -------
    dict | None
        Dict with keys ``link_keys`` and ``total_cost`` if a path exists, otherwise ``None``.
    """

    forbidden = {tuple(flk) for flk in forbidden_links}
    if forbid_reverse:
        forbidden |= {(link[1], link[0], link[2]) for link in list(forbidden)}
    graph = nx.MultiDiGraph()

    for _, row in links_df.iterrows():
        key = tuple(row["link_key"])
        if key in forbidden:
            continue

        weight = None
        if weight_col in row and np.isfinite(row[weight_col]) and row[weight_col] > 0:
            weight = float(row[weight_col])
        elif "length" in row and np.isfinite(row["length"]) and row["length"] > 0:
            weight = float(row["length"])
        else:
            weight = 1.0

        graph.add_edge(
            key[0],
            key[1],
            key=key[2],
            weight=weight,
            link_key=key,
            geometry=row.geometry if hasattr(row, "geometry") else None,
        )

    try:
        node_sequence = nx.algorithms.shortest_paths.weighted.dijkstra_path(
            graph, origin, destination, weight="weight"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    link_keys = []
    total_cost = 0.0
    for u, v in zip(node_sequence[:-1], node_sequence[1:]):
        edge_dict = graph.get_edge_data(u, v)
        if not edge_dict:
            return None
        edge_key, edge_data = min(edge_dict.items(), key=lambda item: item[1].get("weight", 1.0))
        link_keys.append(edge_data.get("link_key", (u, v, edge_key)))
        total_cost += float(edge_data.get("weight", 0.0))

    return {"link_keys": link_keys, "total_cost": total_cost}

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


def add_area_of_interest_layer(nodes_df):
    """
    To add the Bounding Box layer
    """

    # Create a map centered at a specific latitude and longitude
    # Taking the center of the available data
    avg_location = nodes_df[['lat', 'lon']].mean()

        # Central point
    center_latitude = avg_location[0]
    center_longitude = avg_location[1]

    bbox = get_bounding_box(latitude_in_degrees=center_latitude,
                            longitude_in_degrees=center_longitude, 
                            width_km=2, 
                            height_km=5)
    polygon = create_polygon_from_bbox(bbox)

    polygon_coords = list(polygon.exterior.coords)
    polygon_coords = [(lat, lng) for lng, lat in polygon_coords]

    map_city = folium.Map(location=avg_location, zoom_start=13)

    # Create a FeatureGroup for markers/ nodes
    bbox_layer = FeatureGroup(name='Area of interest (AOI)')
    
    folium.Polygon(polygon_coords, 
                   color="darkcyan", 
                   weight=5,
                   fill=True, 
                   fill_color="darkcyan", 
                   fill_opacity=0.1).add_to(bbox_layer)

    # Add the nodes layer to the map
    bbox_layer.add_to(map_city)
    return map_city

def add_nodes_layer(nodes_df, foilum_map):
    """To add the nodes layer
    """

    #Considering the actual nodes data only for plotting in folium map
    actual_nodes_df=nodes_df.loc[nodes_df.type=="real"]
    
    # Create a FeatureGroup for markers/ nodes
    nodes_layer = FeatureGroup(name='Street intersections')
    # map nodes in map
    for _, r in actual_nodes_df.iterrows():
        lat = r['lat']
        lon = r['lon']
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color='darkblue',
            popup = r['key']
        ).add_to(nodes_layer)
    
    # Add the nodes layer to the map
    nodes_layer.add_to(foilum_map)

def add_virtual_links(links_df, folium_map):
    """To add the virtual links layer
    """
    virtual_links_df=links_df.loc[links_df.link_type!="street"]
    # Create a FeatureGroup for actual links
    vlinks_layer = FeatureGroup(name='Connectors')

    # map links in map
    for _, r in virtual_links_df.iterrows():
        sim_geo_edge = gpd.GeoSeries(r["geometry"])
        geo_j_edge = sim_geo_edge.to_json()

        #lanes=[]
        #if type(r.lanes)==int or (type(r.lanes)==str and r.lanes!="inf"):
            #lanes.append(ast.literal_eval(r.lanes))

        pop_up_data=[f'Generator/ Attractor',#{r["link_type"]}',
                     f'Lanes: {r.lanes if math.isinf(r.lanes) else int(r.lanes)}', #{lanes[0][0] if len(lanes)!=0 and type(lanes[0])==str else lanes[0] if len(lanes)!=0 and type(lanes[0])==int else None}',
                     f'Max speed (mph): {round(r["maxspeed_impute"],1)}',
                     f'Capacity: {r["capacity"]}']
        pop_up = "<br>".join(pop_up_data)
        popup_msg=folium.Popup(pop_up, max_width=300)

        geo_j_edge = folium.GeoJson(data=geo_j_edge,
                                    popup=popup_msg,
                                    style_function=lambda x: {"color": "steelblue",
                                                              'weight': 2})
        geo_j_edge.add_to(vlinks_layer)

        # Add the links layer to the map
    vlinks_layer.add_to(folium_map)

def add_links_layer(links_df, folium_map):
    """To add the links layer
    """
    
    # Create a FeatureGroup for actual links
    links_layer = FeatureGroup(name='Street segments')

    # map links in map
    for _, r in links_df.iterrows():
        sim_geo_edge = gpd.GeoSeries(r["geometry"])
        geo_j_edge = sim_geo_edge.to_json()

        pop_up_data=[f'Lanes: {r.lanes if math.isinf(r.lanes) else int(r.lanes)}',#{lanes[0][0] if len(lanes)!=0 and type(lanes[0])==str else lanes[0] if len(lanes)!=0 and type(lanes[0])==int else 1}',
                     f'Max speed (mph): {round(r["maxspeed_impute"],1)}',
                     f'Capacity (veh/hr): {r["capacity"]}']
        pop_up = "<br>".join(pop_up_data)
        popup_msg=folium.Popup(pop_up, max_width=300)

        geo_j_edge = folium.GeoJson(data=geo_j_edge,
                                    popup=popup_msg,
                                    style_function=lambda x: {"color": "gray",'weight': 3})
        geo_j_edge.add_to(links_layer)
    
    # Add the links layer to the map
    links_layer.add_to(folium_map)

def add_toll_links_layer(toll_links_df, folium_map, time):
    """To add toll links layer in the folium map
    """

    # Create a FeatureGroup for toll links
    toll_links_layer = FeatureGroup(name='Tolls')

    for _, r in toll_links_df.iterrows():

        pop_up_data=[f'Toll price($): {str(r["toll_price"])}', 
                     f'Hour of the day (24 hrs format): {time}:00',
                     f'Lanes: {r.lanes if math.isinf(r.lanes) else int(r.lanes)}', #{lanes[0][0] if len(lanes)!=0 and type(lanes[0])==str else lanes[0] if len(lanes)!=0 and type(lanes[0])==int else None}',
                     f'Capacity: {r["capacity"]}',
                     f'Volume capacity ratio (before toll): {round(r["bf_flow"]/r["capacity"],1)}',
                     f'Volume capacity ratio (after toll): {round(r["af_flow"]/r["capacity"],1)}']
        pop_up = "<br>".join(pop_up_data)
        popup=folium.Popup(pop_up, max_width=300)
        folium.Marker(location=[r["lat_u"], r["lon_u"]],
                             popup=popup,
                             icon=folium.Icon(color='black')).add_to(toll_links_layer)
        sim_geo_edge = gpd.GeoSeries(r["geometry"])
        geo_j_edge = sim_geo_edge.to_json()
        geo_j_edge = folium.GeoJson(data=geo_j_edge, style_function=lambda x: {"color": "black",
                                                                               "weight": 6})
        geo_j_edge.add_to(toll_links_layer)

    # Add the toll links layer to the map
    toll_links_layer.add_to(folium_map)

"""Functions for colormaps & style(color, weight etc) for traffic flow & travel speed 
change after intervention."""
def tf_change_colormap(value, v_min, v_max):
    if value < 0:
        return cm.LinearColormap(['green', 'lightgreen'], vmin=v_min, vmax=0)(value)
    elif value > 0:
        return cm.LinearColormap(['orange', 'red'], vmin=0, vmax=v_max)(value)

def speed_change_colormap(value, v_min, v_max):
    if value < 0:
        return cm.LinearColormap(['red', 'orange'], vmin=v_min, vmax=0)(value)
    elif value > 0:
        return cm.LinearColormap(['lightgreen', 'green'], vmin=0, vmax=v_max)(value)

def tf_change_style_function(value, v_min, v_max):
    def style_function(feature):
        return {
            'color': tf_change_colormap(value, v_min, v_max),
            'weight': 6,
            }
    return style_function

def speed_change_style_function(value, v_min, v_max):
    def style_function(feature):
        return {
            'color': speed_change_colormap(value, v_min, v_max),
            'weight': 6,
            }
    return style_function

def add_tf_change_layer(actual_links_df, folium_map):
    """To add the layer for change in traffic flow due to intervention
    """

    # Create a FeatureGroup for changes in treatment from baseline
    changed_flow_layer = FeatureGroup(name='Flow change (after-before)')

    #calculating the min & max value for normalized traffic flow range.
    v_min=(actual_links_df.af_flow-actual_links_df.bf_flow).min()
    v_max=(actual_links_df.af_flow-actual_links_df.bf_flow).max()

    colormap=cm.LinearColormap(
        colors=['green', 'lightgreen', 'orange', 'red'],
        index=[v_min, 0, 0, v_max],
        vmin=v_min,
        vmax=v_max)
    colormap.caption="Flow change"

    for _, r in actual_links_df.iterrows():
        # links with difference in treated & baseline traffic flow values.
        # Intermediate solution as predicted flow is coming 0 for some links.
        if r["af_flow"]-r["bf_flow"]!=0 and r["bf_flow"]!=0:
            flow_change=round(r["af_flow"]-r["bf_flow"],2)
            inc_dec_value="increased" if flow_change>0 else "decreased"
            # External popup data
            
            pop_up_data=[f'Lanes: {r.lanes if math.isinf(r.lanes) else int(r.lanes)}', #lanes[0][0] if len(lanes)!=0 and type(lanes[0])==str else lanes[0] if len(lanes)!=0 and type(lanes[0])==int else 1}',
                         f'Capacity: {r["capacity"]}',
                         f'Traffic flow {inc_dec_value} by: \
                {abs(flow_change)} ({round(abs(flow_change/r["bf_flow"])*100,2)}%).']
            
            popup_msg = "<br>".join(pop_up_data)

            # capping the max as 1 & min as -1
            style_function = tf_change_style_function(flow_change, 
                                                     v_min, 
                                                     v_max)
            sim_geo_edge = gpd.GeoSeries(r["geometry"])
            geo_j_edge = sim_geo_edge.to_json()
            pop_up=folium.Popup(popup_msg, max_width=300)
            geo_j_edge = folium.GeoJson(data=geo_j_edge, 
                                        style_function=style_function,
                                        popup=pop_up)
            geo_j_edge.add_to(changed_flow_layer)

    colormap.add_to(folium_map)
    # Add the nodes layer to the map
    changed_flow_layer.add_to(folium_map)


def get_top_flow_change_links(actual_links_df, top_n=3):
    """Return the links with the largest positive flow increases."""

    if actual_links_df.empty:
        return actual_links_df

    df = actual_links_df.copy(deep=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["bf_flow", "af_flow"])
    if df.empty:
        return df

    df["flow_change"] = df["af_flow"] - df["bf_flow"]
    df = df[df["flow_change"] > 0]
    if df.empty:
        return df

    top_n = max(1, min(top_n, len(df)))
    return df.sort_values("flow_change", ascending=False).head(top_n)


def add_top_flow_change_layer(actual_links_df, folium_map, top_n=3, od_pairs_dict=None, od_paths_dict=None,
                              baseline_paths_dict=None, alternate_paths_dict=None):
    """Highlight the top links experiencing flow increases."""

    top_links_df = get_top_flow_change_links(actual_links_df, top_n)
    if top_links_df.empty:
        return

    top_layer = FeatureGroup(name='Top flow increase links', show=False)

    for _, r in top_links_df.iterrows():
        change = r["flow_change"]
        color = 'red'
        link_key_tuple = tuple(r["link_key"])
        od_pairs = None
        if od_pairs_dict is not None:
            od_pairs = od_pairs_dict.get(link_key_tuple)
        path_geoms_by_od = None
        if od_paths_dict is not None:
            path_geoms_by_od = od_paths_dict.get(link_key_tuple)
        baseline_geoms_by_od = None
        if baseline_paths_dict is not None:
            baseline_geoms_by_od = baseline_paths_dict.get(link_key_tuple)
        alternate_geoms_by_od = None
        if alternate_paths_dict is not None:
            alternate_geoms_by_od = alternate_paths_dict.get(link_key_tuple)
        pop_up_data = [
            f'Link: {r["link_key"]}',
            f'Baseline flow (veh/hr): {round(r["bf_flow"], 1)}',
            f'After flow (veh/hr): {round(r["af_flow"], 1)}',
            f'Change (veh/hr): {change:+.1f}',
        ]
        if od_pairs:
            formatted = ", ".join(f"({orig}, {dest}): {flow:.1f} veh/hr" for orig, dest, flow in od_pairs)
            pop_up_data.append(f'OD usage (after intervention): {formatted}')
        popup_msg = "<br>".join(pop_up_data)

        sim_geo_edge = gpd.GeoSeries(r["geometry"])
        geo_j_edge = sim_geo_edge.to_json()
        geo_json = folium.GeoJson(
            data=geo_j_edge,
            style_function=lambda x, col=color: {"color": col, "weight": 8, "opacity": 0.9},
        )
        geo_json.add_child(folium.Popup(popup_msg, max_width=300))
        geo_json.add_to(top_layer)

        if path_geoms_by_od:
            color_cycle = ['orange', 'blue', 'purple', 'green']
            for od_idx, (od_pair, path_entries) in enumerate(path_geoms_by_od.items(), start=1):
                if not path_entries:
                    continue
                od_group = FeatureGroupSubGroup(
                    top_layer,
                    name=f"OD {od_pair[0]}→{od_pair[1]}",
                    show=False,
                )
                od_group.add_to(folium_map)

                baseline_entries = [] if baseline_geoms_by_od is None else baseline_geoms_by_od.get(od_pair, [])
                for base_idx, (base_flow, base_geom) in enumerate(baseline_entries):
                    if base_geom is None or base_geom.is_empty:
                        continue
                    base_group = FeatureGroupSubGroup(
                        od_group,
                        name=f"Baseline path {base_idx + 1} ({base_flow:.1f} veh/hr)",
                        show=False,
                    )
                    base_group.add_to(folium_map)
                    sequences = [list(base_geom.coords)] if base_geom.geom_type == "LineString" else [list(geom.coords) for geom in base_geom.geoms]
                    for coords in sequences:
                        if len(coords) < 2:
                            continue
                        latlon = [(coord[1], coord[0]) for coord in coords]
                        folium.PolyLine(
                            locations=latlon,
                            color='gray',
                            weight=4,
                            opacity=0.4,
                            dash_array='6,6',
                        ).add_to(base_group)

                main_flow, main_geom = path_entries[0]
                if main_geom is not None and not main_geom.is_empty:
                    seqs = [list(main_geom.coords)] if main_geom.geom_type == "LineString" else [list(geom.coords) for geom in main_geom.geoms]
                    for coords in seqs:
                        if len(coords) < 2:
                            continue
                        latlon = [(coord[1], coord[0]) for coord in coords]
                        folium.PolyLine(
                            locations=latlon,
                            color=color_cycle[0],
                            weight=5,
                            opacity=0.7,
                        ).add_to(od_group)

                for alt_idx, (alt_flow, alt_geom) in enumerate(path_entries[1:], start=1):
                    if alt_geom is None or alt_geom.is_empty:
                        continue
                    alt_group = FeatureGroupSubGroup(
                        od_group,
                        name=f"Path {alt_idx} ({alt_flow:.1f} veh/hr)",
                        show=False,
                    )
                    alt_group.add_to(folium_map)
                    sequences = [list(alt_geom.coords)] if alt_geom.geom_type == "LineString" else [list(geom.coords) for geom in alt_geom.geoms]
                    for coords in sequences:
                        if len(coords) < 2:
                            continue
                        latlon = [(coord[1], coord[0]) for coord in coords]
                        folium.PolyLine(
                            locations=latlon,
                            color=color_cycle[(alt_idx) % len(color_cycle)],
                            weight=4,
                            opacity=0.6,
                        ).add_to(alt_group)

                if alternate_geoms_by_od:
                    alt_entry = alternate_geoms_by_od.get(od_pair)
                    if alt_entry and alt_entry.get("geometry") is not None:
                        cost = alt_entry.get("total_cost")
                        suffix = f" [{cost:.1f} s]" if cost else ""
                        comp_group = FeatureGroupSubGroup(
                            od_group,
                            name=f"Alt path (avoid link){suffix}",
                            show=False,
                        )
                        comp_group.add_to(folium_map)
                        alt_geom = alt_entry["geometry"]
                        sequences = [list(alt_geom.coords)] if alt_geom.geom_type == "LineString" else [list(geom.coords) for geom in alt_geom.geoms]
                        for coords in sequences:
                            if len(coords) < 2:
                                continue
                            latlon = [(coord[1], coord[0]) for coord in coords]
                            folium.PolyLine(
                                locations=latlon,
                                color='teal',
                                weight=4,
                                opacity=0.65,
                            ).add_to(comp_group)

    top_layer.add_to(folium_map)


def add_abs_speed_change_layer(actual_links_df, folium_map):
    """To add the layer for change in speed due to intervention
    """

    # Create a FeatureGroup for changes in treatment from baseline
    changed_speed_layer = FeatureGroup(name='Speed change (after-before)')

    #calculating the min & max value for speed change range.
    v_min=(actual_links_df.af_speed - actual_links_df.bf_speed).min()*2.24
    v_max=(actual_links_df.af_speed - actual_links_df.bf_speed).max()*2.24
    
    colormap=cm.LinearColormap(
        colors=['red', 'orange', 'lightgreen', 'green'],
        index=[v_min, 0, 0, v_max],
        vmin=v_min,
        vmax=v_max)
    colormap.caption="speed change (mph)"
    
    for _, r in actual_links_df.iterrows():
        # 2.24 to convert m/sec to miles/hr
        bf_speed=round(r["bf_speed"]*2.24,2)
        af_speed=round(r["af_speed"]*2.24,2)
        # links with difference in treated & baseline speed values.
        if af_speed-bf_speed!=0:
            speed_change=round((af_speed-bf_speed),1)
            inc_dec_value="increased" if speed_change>0 else "decreased"

            free_flow_tt=(r["length"]/r["maxspeed_impute"]*0.45) # miles/hr to m/seconds
            tt_delay=round((r["bf_tt"]-free_flow_tt)/free_flow_tt,2)

            pop_up_data=[f'Free flow speed (mph): {round(r["maxspeed_impute"],1)}',
                         f'Travel time delay factor: {tt_delay}',
                         f'Speed (mph) {inc_dec_value} by: \
                    {abs(speed_change)} ({round(abs(speed_change/bf_speed)*100,1)}%)']
            popup_msg = "<br>".join(pop_up_data)

            # capping the max as 1 & min as -1
            style_function = speed_change_style_function(speed_change, 
                                                     v_min, 
                                                     v_max)
            sim_geo_edge = gpd.GeoSeries(r["geometry"])
            geo_j_edge = sim_geo_edge.to_json()
            pop_up=folium.Popup(popup_msg, max_width=300)
            geo_j_edge = folium.GeoJson(data=geo_j_edge, 
                                        style_function=style_function,
                                        popup=pop_up)
            geo_j_edge.add_to(changed_speed_layer)
            
    colormap.add_to(folium_map)
    # Add the nodes layer to the map
    changed_speed_layer.add_to(folium_map)

"""Functions for colormaps & style(color, weight etc) for baseline & treated traffic flow."""
def flow_colormap(value, v_min, v_max):
    if value < 0.5:
        return cm.LinearColormap(['green', 'lightgreen'], vmin=0, vmax=0.5)(value)
    elif value > 0.5:
        return cm.LinearColormap(['orange', 'red'], vmin=0.5, vmax=1)(value)

def generate_flow_style_function(value, v_min, v_max):
    def style_function(feature):
        # You can use custom_param in your style logic if needed
        return {
            'color': flow_colormap(value, v_min, v_max),
            'weight': 3,
            }
    return style_function

def add_baseline_tf_layer(actual_links_df, traffic_counter_df, folium_map):
    """To add the layer for baseline traffic flow layer
    """

    # Create a FeatureGroup for baseline traffic flow
    basel_tf_layer = FeatureGroup(name='Flow before intervention')
    
    #calculating the min & max value for normalized traffic flow range.
    v_min=0
    v_max=1
    
    # Define a color map
    colormap=cm.LinearColormap(
        colors=['green', 'lightgreen', 'orange', 'red'],
        index=[0, 0.5, 0.5, 1],
        vmin=v_min,
        vmax=v_max)
    
    for _, r in actual_links_df.iterrows():
        flow=r["bf_flow"]/r["capacity"]

        pop_up_data=[f'Estimated traffic flow (veh/hr): {str(round(r["bf_flow"],1))}',
                     f'Normalized traffic flow (veh/hr): {str(round(flow,1))}', 
                     f'Lanes: {r.lanes if math.isinf(r.lanes) else int(r.lanes)}', #{lanes[0][0] if len(lanes)!=0 and type(lanes[0])==str else lanes[0] if len(lanes)!=0 and type(lanes[0])==int else 1}',
                     f'Capacity: {r["capacity"]}']
        popup_msg = "<br>".join(pop_up_data)

        # capping the max as 1
        style_function = generate_flow_style_function(1 if flow>=1 else flow, 
                                                     v_min, 
                                                     v_max)

        #Keeping threshold of 0.2 (The links with traffic flow more than 0.2 would be highlighted) 
        #if flow>0.2:
        sim_geo_edge = gpd.GeoSeries(r["geometry"])
        geo_j_edge = sim_geo_edge.to_json()
        pop_up=folium.Popup(popup_msg, max_width=300)
        geo_j_edge = folium.GeoJson(data=geo_j_edge, 
                                        style_function=style_function,
                                        popup=pop_up)
        geo_j_edge.add_to(basel_tf_layer)


    # Add the layer to the map
    basel_tf_layer.add_to(folium_map)

def add_treated_tf_layer(actual_links_df, new_links_df=None, folium_map=None):
    """To add the layer for treated traffic flow layer
    """

    # Create a FeatureGroup for treated traffic flow
    treated_tf_layer = FeatureGroup(name='Flow after intervention')

    #calculating the min & max value for normalized traffic flow range.
    v_min=0
    v_max=1
    
    # Define a color map
    colormap=cm.LinearColormap(
        colors=['green', 'lightgreen', 'orange', 'red'],
        index=[0, 0.5, 0.5, 1],
        vmin=v_min,
        vmax=v_max)
    colormap.caption="Normalized Traffic Flow (veh/hr)"
    
    #In case of toll as we dont have any new links-df with capacity changes.
    if new_links_df is None:
        new_links_df=actual_links_df.copy(deep=True)

    for i, r in actual_links_df.iterrows():
        flow=r["af_flow"]/(r["capacity"])

        pop_up_data=[f'Estimated traffic flow (veh/hr): {str(round(r["af_flow"], 1))}',
                     f'Normalized traffic flow (veh/hr): {str(round(flow,1))}', 
                     f'Lanes: {r.lanes if math.isinf(r.lanes) else int(r.lanes)}', #{lanes[0][0] if len(lanes)!=0 and type(lanes[0])==str else lanes[0] if len(lanes)!=0 and type(lanes[0])==int else 1}',
                     # Changed capacity after partial closure or cap enhancement
                     f'Capacity: {new_links_df.loc[i,"capacity"]}']
        popup_msg = "<br>".join(pop_up_data)

        # capping the max as 1
        style_function = generate_flow_style_function(1 if flow>=1 else flow, 
                                                     v_min, 
                                                     v_max)

        #Keeping threshold of 0.2 (The links with traffic flow more than 0.2 would be highlighted) 
        #if flow>0.2:
        sim_geo_edge = gpd.GeoSeries(r["geometry"])
        geo_j_edge = sim_geo_edge.to_json()
        pop_up=folium.Popup(popup_msg, max_width=300)
        geo_j_edge = folium.GeoJson(data=geo_j_edge, 
                                        style_function=style_function,
                                        popup=pop_up)
        geo_j_edge.add_to(treated_tf_layer)

    colormap.add_to(folium_map)
    # Add the layer to the map
    treated_tf_layer.add_to(folium_map)

# Considering absolute speed for speed layer
"""Functions for colormaps & style(color, weight etc) for baseline & treated travel speed."""
def speed_colormap(value, v_min, v_max):
    if value < v_max-((v_max-v_min)/2):
        return cm.LinearColormap(['red', 'orange'], vmin=v_min, vmax=v_max-((v_max-v_min)/2))(value)
    elif value >=v_max-((v_max-v_min)/2):
        return cm.LinearColormap(['lightgreen', 'green'], vmin=v_max-((v_max-v_min)/2), vmax=v_max)(value)
    
def generate_speed_style_function(value, v_min, v_max):
    def style_function(feature):
        # You can use custom_param in your style logic if needed
        return {
            'color': speed_colormap(value, v_min, v_max),
            'weight': 3,
            }
    return style_function

def add_baseline_abs_speed_layer(actual_links_df, traffic_counter_df, folium_map):
    """To add the layer for baseline travel time/speed layer
    """

    # Create a FeatureGroup for baseline travel time
    basel_speed_layer = FeatureGroup(name='Speed before intervention')
    
    v_min=(actual_links_df["bf_speed"]*2.24).min()
    v_max=(actual_links_df["bf_speed"]*2.24).max()
    
    for _, r in actual_links_df.iterrows():
        bf_speed= r['bf_speed']*2.24
        bf_speed_norm= bf_speed/r['maxspeed_impute']

        free_flow_tt=(r["length"]/(r["maxspeed_impute"]*0.45)) # miles/hr to m/seconds
        tt_delay=round((r["bf_tt"]-free_flow_tt)/free_flow_tt,2)

        pop_up_data=[f'Estimated speed (mph): {str(round(bf_speed,1))}',
                     f'Normalized speed: {str(round(bf_speed_norm,1))}', 
                     f'Free flow speed (mph): {round(r["maxspeed_impute"],1)}',
                     f'Travel time delay factor: {tt_delay}']
        popup_msg = "<br>".join(pop_up_data)

        # capping the max as 1
        style_function = generate_speed_style_function(bf_speed,
                                                     v_min, 
                                                     v_max)
        #Deciding a threshold to display links with normalized speed
        #if bf_speed_norm<=1.01*(v_max-((v_max-v_min)/2)) :
        sim_geo_edge = gpd.GeoSeries(r["geometry"])
        geo_j_edge = sim_geo_edge.to_json()
        pop_up=folium.Popup(popup_msg, max_width=300)
        geo_j_edge = folium.GeoJson(data=geo_j_edge, 
                                        style_function=style_function,
                                        popup=pop_up)
        geo_j_edge.add_to(basel_speed_layer)

    # Add the nodes layer to the map
    basel_speed_layer.add_to(folium_map)


def add_treated_abs_speed_layer(actual_links_df, folium_map):
    """To add the layer for treated travel time/speed layer
    """

    # Create a FeatureGroup for baseline traffic flow
    treated_speed_layer = FeatureGroup(name='Speed after intervention')
    
    v_min=(actual_links_df["af_speed"]*2.24).min()
    v_max=(actual_links_df["af_speed"]*2.24).max()

    # Define a color map
    colormap=cm.LinearColormap(
        colors=['red', 'orange', 'lightgreen', 'green'],
        index=[v_min, v_max-((v_max-v_min)/2), v_max-((v_max-v_min)/2), v_max],
        vmin=v_min,
        vmax=v_max)

    colormap.caption="Speed (mph)"
    
    for _, r in actual_links_df.iterrows():
        #af_speed_norm= (r['af_speed']*2.24)/(r['maxspeed_impute'])
        af_speed=r['af_speed']*2.24
        af_speed_norm= af_speed/r['maxspeed_impute']

        free_flow_tt=(r["length"]/(r["maxspeed_impute"]*0.45)) # miles/hr to m/seconds
        tt_delay=round((r["af_tt"]-free_flow_tt)/free_flow_tt,2)

        pop_up_data=[f'Estimated speed (mph): {str(round(af_speed,1))}',
                     f'Normalized speed: {str(round(af_speed_norm,1))}', 
                     f'Free flow speed (mph): {round(r["maxspeed_impute"],1)}',
                     f'Travel time delay factor: {tt_delay}']
        popup_msg = "<br>".join(pop_up_data)
        
        # capping the max as 1
        style_function = generate_speed_style_function(af_speed,
                                                     v_min, 
                                                     v_max)
        #Deciding a threshold to display links with normalized speed
        #if af_speed_norm<=1.01*(v_max-((v_max-v_min)/2)) :
        sim_geo_edge = gpd.GeoSeries(r["geometry"])
        geo_j_edge = sim_geo_edge.to_json()
        pop_up=folium.Popup(popup_msg, max_width=300)
        geo_j_edge = folium.GeoJson(data=geo_j_edge, 
                                        style_function=style_function,
                                        popup=pop_up)
        geo_j_edge.add_to(treated_speed_layer)

    colormap.add_to(folium_map)
    # Add the nodes layer to the map
    treated_speed_layer.add_to(folium_map)

def add_traffic_counters_layer(traffic_counter_df, links_df, time, folium_map):
    """To add the layer of Traffic counters.
    """

    # Create a FeatureGroup for traffic counters
    traffic_counters_layer = FeatureGroup(name='Traffic counters')
    # Check only for a particular time for now.
    #traffic_counter_df=traffic_counter_df.drop_duplicates(subset=["link_key"], keep="last")

    traffic_counter_links=links_df.loc[links_df.link_keys_osmids.isin(traffic_counter_df["link_key"].values)]

    for _, r in traffic_counter_df.iterrows():
        matching = traffic_counter_links.loc[traffic_counter_links.link_keys_osmids == r["link_key"]]
        if matching.empty:
            continue
        if matching["observed_flow"].iloc[0] == 0:
            continue

        lat = r['Latitude']
        lon = r['Longitude']
        pop_up_data=[f'PeMS id: {r["Station"]}',
                     f'Hour of the day (24 hrs format): {time}:00',
                     f'Traffic flow (veh/hr): {r["Total Flow"]}',
                     f'Average speed (mph): {round(r["Avg Speed"],1)}'
                     ]
        popup_msg = "<br>".join(pop_up_data)
        pop_up=folium.Popup(popup_msg, max_width=300)
        folium.Marker(
            location=[lat, lon],
            radius=3,
            icon=folium.Icon(color='purple'),
            popup = pop_up
        ).add_to(traffic_counters_layer)

    
    for _,r in traffic_counter_links.iterrows():
        if r["observed_flow"]>0:
            obs_flow_error=(abs(r["bf_flow"]-r["observed_flow"])/r["observed_flow"])*100
            obs_speed=round((r["length"]/r["observed_tt"])*2.24,1)
            pred_speed=round((r["length"]/r["bf_tt"])*2.24,1)
            obs_speed_error=(abs(pred_speed-obs_speed)/ obs_speed)*100

            pop_up_data=[f'Hour of the day (24 hrs format): {time}:00',
                        f'Observed traffic flow (veh/hr): {str(round(r["observed_flow"],1))}', 
                        f'Estimated traffic flow (veh/hr): {round(r["bf_flow"],1)}',
                        f'Traffic flow error (%): {round(obs_flow_error,2)}',
                        f'Observed speed (mph): {obs_speed}',
                        f'Estimated speed (mph): {pred_speed}',
                        f'Speed error (%): {round(obs_speed_error,2)}']
            
            popup_msg = "<br>".join(pop_up_data)

            sim_geo_edge = gpd.GeoSeries(r["geometry"])
            geo_j_edge = sim_geo_edge.to_json()
            pop_up=folium.Popup(popup_msg, max_width=300)
            geo_j_edge = folium.GeoJson(data=geo_j_edge, 
                                        style_function=lambda x: {"color": "purple",
                                                                "weight": 5},
                                        popup=pop_up)
            geo_j_edge.add_to(traffic_counters_layer)
        # Intermediate solution as observed flow is coming 0 for some links.

    # Add the nodes layer to the map
    traffic_counters_layer.add_to(folium_map)

def add_taz_layer(taz_df, folium_map):
    """To add the layer of TAZ.
    """

    # Create a FeatureGroup for traffic counters
    taz_layer = FeatureGroup(name='Traffic analysis zones (TAZ)')

    if "centroid" not in taz_df.columns:
        taz_df = taz_df.copy()
        taz_df["centroid"] = taz_df.geometry.centroid

    # Dropping duplicate data based on centroid
    taz_df = taz_df[~taz_df["centroid"].duplicated(keep='first')]

    for _, r in taz_df.iterrows():
        sim_geo = gpd.GeoSeries(r["geometry_polygon"])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {"fillColor": "lightblue",
                                                                     "fillOpacity": 0.5, #0.0001
                                                                     "opacity": 1.0})
        geo_j.add_to(taz_layer)

    # Add the TAZ layer to the map
    taz_layer.add_to(folium_map)

def calculate_flow(links_df):
    """To calculate the trips/ flow generated/ attracted from virtual nodes.
    
    """
    g_vlinks_df=links_df.loc[links_df.link_type=="generator"]
    a_vlinks_df=links_df.loc[links_df.link_type=="attractor"]
    g_vlinks_df["link_key"]=[x[0] for x in g_vlinks_df.link_key]
    a_vlinks_df["link_key"]=[x[1] for x in a_vlinks_df.link_key]
    g_vlinks_grp_df=g_vlinks_df[["link_key", "bf_flow"]].groupby("link_key").sum().reset_index()
    a_vlinks_grp_df=a_vlinks_df[["link_key", "bf_flow"]].groupby("link_key").sum().reset_index()
    gen_flow_dict = g_vlinks_grp_df.set_index('link_key')['bf_flow'].to_dict()
    attr_flow_dict = a_vlinks_grp_df.set_index('link_key')['bf_flow'].to_dict()
    return gen_flow_dict, attr_flow_dict


def add_vnodes_layer(nodes_df, links_df, ref_g_gen, folium_map):
    """To add the layer of virtual nodes (attractors & generators).
    """

    # Create a FeatureGroup for traffic counters
    gen_attr_layer = FeatureGroup(name='Trip generators/attractors')

    #Considering the actual nodes data only for plotting in folium map
    virtual_nodes_df=nodes_df.loc[nodes_df.type!="real"]

    geo_grouped_df=virtual_nodes_df.groupby(["lon", "lat"]).count().reset_index()
    gen_flow_dict, attr_flow_dict= calculate_flow(links_df)
    
    origin_nodes=[x[0] for x in links_df.link_key]
    dest_nodes=[x[1] for x in links_df.link_key]
    origin_count_dict = Counter(origin_nodes)
    dest_count_dict=Counter(dest_nodes)
    print(gen_flow_dict)
    print(attr_flow_dict)
    for _, r in geo_grouped_df.iterrows():
        lat = r['lat']
        lon = r['lon']
        node_data=virtual_nodes_df.loc[(virtual_nodes_df.lon==lon) & (virtual_nodes_df.lat==lat)]
        gen_node_key=node_data.loc[node_data.type=="generator"]["key"].values[0]
        attr_node_key=node_data.loc[node_data.type=="attractor"]["key"].values[0]
        pop_up_msg=[f'Estimated trip generation (veh/hr): {round(gen_flow_dict[gen_node_key],2)}',
                    f'Estimated trip attraction (veh/hr): {round(attr_flow_dict[attr_node_key],2)}',
                    f'Reference trip generation (veh/hr): {round(ref_g_gen[gen_node_key],2)}',
                    f'Generator links: {origin_count_dict[gen_node_key]}',
                    f'Attractor links: {dest_count_dict[attr_node_key]}',
                    ]
        pop_up="<br>".join(pop_up_msg)
        popup_msg=folium.Popup(pop_up, max_width=300)
        folium.CircleMarker(
            location=[lat, lon],
            radius=2.5, 
            color="steelblue",
            popup=popup_msg).add_to(gen_attr_layer)

    # Add the TAZ layer to the map
    gen_attr_layer.add_to(folium_map)


def add_road_closure_links_layer(closure_links_df, folium_map):
    
    """To add closure links layer in the folium map
    """

    # Create a FeatureGroup for toll links
    closure_links_layer = FeatureGroup(name='Road closures')
    for _, r in closure_links_df.iterrows():
        af_flow=r["af_flow"]
        bf_flow=r["bf_flow"]
        if bf_flow!=0:
            flow_change_norm=(af_flow-bf_flow)/bf_flow
        else:
            flow_change_norm=1

        # COnvert speed from meters/sec to miles/hr
        af_speed=(r["length"]/r["af_tt"])*2.24
        bf_speed=(r["length"]/r["bf_tt"])*2.24
        speed_change=abs(af_speed-bf_speed)/bf_speed

        #lanes=[]
        #if type(r.lanes)==int or (type(r.lanes)==str and r.lanes!="inf"):
            #lanes.append(ast.literal_eval(r.lanes))
        
        pop_up_data=[f'Road Closure Link: {r["link_key"][:2]}',
                     f'Lanes: {r.lanes if math.isinf(r.lanes) else int(r.lanes)}', #{lanes[0][0] if len(lanes)!=0 and type(lanes[0])==str else lanes[0] if len(lanes)!=0 and type(lanes[0])==int else None}',
                     f'Capacity: {r["capacity"]}',]
                     #f'Volume capacity ratio (before road closure): {round(r["bf_flow"]/r["capacity"],2)}',
                     #f'Volume capacity ratio (after road closure): {round(r["af_flow"]/r["capacity"],2)}']

        pop_up = "<br>".join(pop_up_data)
        popup_msg=folium.Popup(pop_up, max_width=300)
        # Showing popup on middle of the closure link.
        folium.Marker(location=[(r["lat_u"]), (r["lon_u"])],
                      #[(r["lat_u"]+r["lat_v"])/2, (r["lon_u"]+r["lon_v"])/2],
                             popup=popup_msg,
                             icon=folium.Icon(color='black')).add_to(closure_links_layer)

        pop_up_data=[f'Lanes: {r.lanes if math.isinf(r.lanes) else int(r.lanes)}', #{lanes[0][0] if len(lanes)!=0 and type(lanes[0])==str else lanes[0] if len(lanes)!=0 and type(lanes[0])==int else None}',
                     f'Capacity: {r["capacity"]}',
                     f'Max speed (mph): {round(r["maxspeed_impute"],1)}',
                     f'Change in flow (veh/hr): {round(af_flow-bf_flow,1)}({round(abs(flow_change_norm)*100,1)}%)', 
                     f'Change in speed (mph): {round(af_speed-bf_speed,1)}({round(abs(speed_change)*100,1)}%)']
        
        pop_up = "<br>".join(pop_up_data)
        popup_msg=folium.Popup(pop_up, max_width=300)
        sim_geo_edge = gpd.GeoSeries(r["geometry"])
        geo_j_edge = sim_geo_edge.to_json()
        geo_j_edge = folium.GeoJson(data=geo_j_edge,
                                    popup=popup_msg,
                                    style_function=lambda x: {"color": "black",
                                                              "weight": 5})
        geo_j_edge.add_to(closure_links_layer)

    # Add the toll links layer to the map
    closure_links_layer.add_to(folium_map)

"""Functions for colormaps & style(color, weight etc) for old & new paths after road closure"""
def new_paths_colormap(value, v_min, v_max):
    #if value < v_max:
    return cm.LinearColormap(['orange', 'orangered', 'red'], vmin=v_min, vmax=v_max)(value)
    #elif value >= v_max/2:
        #return cm.LinearColormap(['lightred', 'red'], vmin=v_max/2, vmax=v_max)(value)

def generate_paths_style_function(value, v_min, v_max, old_paths_flag):
    def style_function(feature):
        # We can use custom_param in your style logic if needed
        if old_paths_flag:
            return {
                'color': "brown", #flow_colormap(value, v_min, v_max),
                'weight': 5*1.25,
            }
        else:
            return {
                'color': "lightseagreen", #new_paths_colormap(value, v_min, v_max),
                'weight': 5,
            }
    return style_function

def add_affected_od_old_paths_layer(links_df, closure_links_df, nodes_df, old_links_key_list,map_city):

    """To add the layer for road clousure for affected ODs existing paths.
    """
    # Create a FeatureGroup for baseline traffic flow
    old_paths_layer = FeatureGroup(name='Streets affected in origin-destination pair')
    
    affected_links_df=links_df.loc[links_df.link_key.isin(old_links_key_list)]

    #calculating the min & max value for normalized traffic flow range.
    v_min=affected_links_df["bf_flow"].min()
    v_max=affected_links_df["bf_flow"].max()
    # Excluding the closure link as it is managed in road clousre link layer.
    affected_links_df=affected_links_df.loc[~affected_links_df.link_key.isin(closure_links_df["link_key"])]
    
    for _, r in affected_links_df.iterrows():
        af_flow=r["af_flow"]
        bf_flow=r["bf_flow"]
        if bf_flow!=0:
            flow_change_norm=(af_flow-bf_flow)/bf_flow
        else:
            flow_change_norm=1
        # COnvert speed from meters/sec to miles/hr
        af_speed=(r["length"]/r["af_tt"])*2.24
        bf_speed=(r["length"]/r["bf_tt"])*2.24
        speed_change=(af_speed-bf_speed)/bf_speed

        pop_up_data=[f'Change in flow (veh/hr): {round(af_flow-bf_flow,1)}({round(abs(flow_change_norm)*100,1)}%)', 
                     f'Change in speed (mph): {round(af_speed-bf_speed,1)}({round(abs(speed_change)*100,1)}%)']
        
        popup_msg = "<br>".join(pop_up_data)

        style_function = generate_paths_style_function(flow_change_norm, v_min, v_max, True)

        sim_geo_edge = gpd.GeoSeries(r["geometry"])
        geo_j_edge = sim_geo_edge.to_json()
        pop_up=folium.Popup(popup_msg, max_width=300)
        geo_j_edge = folium.GeoJson(data=geo_j_edge, 
                                    style_function=style_function,
                                    popup=pop_up)
        geo_j_edge.add_to(old_paths_layer)

    #Origin node mapping
    origin_node=nodes_df.loc[nodes_df.key==old_links_key_list[0][0]]
    lat = origin_node['lat']
    lon = origin_node['lon']
    folium.Marker(
        location=[lat, lon],
        radius=4,
        icon=folium.Icon(color='darkblue'),
        popup = f'Origin: {origin_node["key"].values[0]}'
    ).add_to(old_paths_layer)

    #Destination node mapping
    dest_node=nodes_df.loc[nodes_df.key==old_links_key_list[-1][1]]
    lat = dest_node['lat']
    lon = dest_node['lon']
    folium.Marker(
        location=[lat, lon],
        radius=5,
        icon=folium.Icon(color='green'),
        popup = f'Destination: {dest_node["key"].values[0]}'
    ).add_to(old_paths_layer)

    # Add the layer to the map
    old_paths_layer.add_to(map_city)

def add_affected_od_new_paths_layer(links_df, nodes_df, new_links_key_list, map_city):

    """To add the road closure layer for affected ODs new paths.
    """
    # Create a FeatureGroup for baseline traffic flow
    new_paths_layer = FeatureGroup(name='Rerouting in affected origin-destination pair')
    
    affected_links_df=links_df.loc[links_df.link_key.isin(new_links_key_list)]

    #calculating the min & max value for normalized traffic flow range.
    v_min=affected_links_df["bf_flow"].min()
    v_max=affected_links_df["bf_flow"].max()
    
    for _, r in affected_links_df.iterrows():
        af_flow=r["af_flow"]
        bf_flow=r["bf_flow"]
        if bf_flow!=0:
            flow_change_norm=(af_flow-bf_flow)/bf_flow
        else:
            flow_change_norm=1
        # Convert speed from meters/sec to miles/hr
        af_speed=(r["length"]/r["af_tt"])*2.24
        bf_speed=(r["length"]/r["bf_tt"])*2.24
        speed_change=(af_speed-bf_speed)/bf_speed

        pop_up_data=[f'Traffic flow (after) (veh/hr): {round(af_flow,1)}',
                     f'Speed (after) (mph): {round(af_speed,1)}',
                     f'Change in flow (veh/hr): {round(af_flow-bf_flow,1)}({round(abs(flow_change_norm)*100,1)}%)', 
                     f'Change in speed (mph): {round(af_speed-bf_speed,1)}({round(abs(speed_change)*100,1)}%)']
        
        popup_msg = "<br>".join(pop_up_data)
        # capping the max as 1
        style_function = generate_paths_style_function(flow_change_norm, v_min, v_max, False)

        sim_geo_edge = gpd.GeoSeries(r["geometry"])
        geo_j_edge = sim_geo_edge.to_json()
        pop_up=folium.Popup(popup_msg, max_width=300)
        geo_j_edge = folium.GeoJson(data=geo_j_edge, 
                                    style_function=style_function,
                                    popup=pop_up)
        geo_j_edge.add_to(new_paths_layer)

    #Origin node mapping
    origin_node=nodes_df.loc[nodes_df.key==new_links_key_list[0][0]]
    lat = origin_node['lat']
    lon = origin_node['lon']
    folium.Marker(
        location=[lat, lon],
        radius=4,
        icon=folium.Icon(color='darkblue'),
        popup = f'Origin: {origin_node["key"].values[0]}'
    ).add_to(new_paths_layer)

    #Destination node mapping
    dest_node=nodes_df.loc[nodes_df.key==new_links_key_list[-1][1]]
    lat = dest_node['lat']
    lon = dest_node['lon']
    folium.Marker(
        location=[lat, lon],
        radius=5,
        icon=folium.Icon(color='green'),
        popup = f'Destination: {dest_node["key"].values[0]}'
    ).add_to(new_paths_layer)

    # Add the layer to the map
    new_paths_layer.add_to(map_city)

def add_traversed_links_layer(links_df, paths, map_city):

    """To add a layer to check the traversed links in paths. 
    """
    traversed_links={}
    for path in paths:
        for link in path.links:
            if link.key not in traversed_links.keys():
                traversed_links[link.key]=1
            else:
                traversed_links[link.key]+=1


     # Create a FeatureGroup for traversed links layer
    traversed_links_layer = FeatureGroup(name='Streets traversed by paths')

    traversed_links_df=links_df.loc[links_df.link_key.isin(traversed_links.keys())]

    for _, r in traversed_links_df.iterrows():
        pop_up_data=[f'Traversed in paths frequency: {traversed_links[r["link_key"]]}']
        pop_up = "<br>".join(pop_up_data)
        popup_msg=folium.Popup(pop_up, max_width=300)
        sim_geo_edge = gpd.GeoSeries(r["geometry"])
        geo_j_edge = sim_geo_edge.to_json()
        geo_j_edge = folium.GeoJson(data=geo_j_edge,
                                    popup=popup_msg,
                                    style_function=lambda x: {"color": "chocolate",
                                                              "weight": 4})
        geo_j_edge.add_to(traversed_links_layer)

    # Add the toll links layer to the map
    traversed_links_layer.add_to(map_city)

def add_road_enhancement_links_layer(enhanced_links_df,  map_city):
    """To add road enhancement links layer in the folium map
    """

    # Create a FeatureGroup for toll links
    enhanced_links_layer = FeatureGroup(name='Road enhancement')
    for _, r in enhanced_links_df.iterrows():
        af_flow=r["af_flow"]
        bf_flow=r["bf_flow"]
        if bf_flow!=0:
            flow_change_norm=(af_flow-bf_flow)/bf_flow
        else:
            flow_change_norm=1

        # COnvert speed from meters/sec to miles/hr
        af_speed=(r["length"]/r["af_tt"])*2.24
        bf_speed=(r["length"]/r["bf_tt"])*2.24
        speed_change=abs(af_speed-bf_speed)/bf_speed

        #lanes=[]
        #if type(r.lanes)==int or (type(r.lanes)==str and r.lanes!="inf"):
            #lanes.append(ast.literal_eval(r.lanes))
        
        pop_up_data=[f'Road enhancement Link: {r["link_key"][:2]}',
                     f'Lanes: {r.lanes if math.isinf(r.lanes) else int(r.lanes)}', #{lanes[0][0] if len(lanes)!=0 and type(lanes[0])==str else lanes[0] if len(lanes)!=0 and type(lanes[0])==int else None}',
                     f'Capacity: {r["capacity"]}',]
                     #f'Volume capacity ratio (before road closure): {round(r["bf_flow"]/r["capacity"],2)}',
                     #f'Volume capacity ratio (after road closure): {round(r["af_flow"]/r["capacity"],2)}']

        pop_up = "<br>".join(pop_up_data)
        popup_msg=folium.Popup(pop_up, max_width=300)
        # Showing popup on middle of the closure link.
        folium.Marker(location=[(r["lat_u"]), (r["lon_u"])],
                      #[(r["lat_u"]+r["lat_v"])/2, (r["lon_u"]+r["lon_v"])/2],
                             popup=popup_msg,
                             icon=folium.Icon(color='black')).add_to(enhanced_links_layer)

        pop_up_data=[f'Lanes: {r.lanes if math.isinf(r.lanes) else int(r.lanes)}', #{lanes[0][0] if len(lanes)!=0 and type(lanes[0])==str else lanes[0] if len(lanes)!=0 and type(lanes[0])==int else None}',
                     f'Capacity: {r["capacity"]}',
                     f'Max speed (mph): {round(r["maxspeed_impute"],1)}',
                     f'Change in flow (veh/hr): {round(af_flow-bf_flow,1)}({round(abs(flow_change_norm)*100,1)}%)', 
                     f'Change in speed (mph): {round(af_speed-bf_speed,1)}({round(abs(speed_change)*100,1)}%)']
        
        pop_up = "<br>".join(pop_up_data)
        popup_msg=folium.Popup(pop_up, max_width=300)
        sim_geo_edge = gpd.GeoSeries(r["geometry"])
        geo_j_edge = sim_geo_edge.to_json()
        geo_j_edge = folium.GeoJson(data=geo_j_edge,
                                    popup=popup_msg,
                                    style_function=lambda x: {"color": "black",
                                                              "weight": 5})
        geo_j_edge.add_to(enhanced_links_layer)

    # Add the road enhanced links layer to the map
    enhanced_links_layer.add_to(map_city)

def perform_toll_interv_visualization(links_df, nodes_df, traffic_counter_df, taz_df, time, vlinks_flag, paths, ref_g_gen):
    """To perform visualization of data in folium map.

    Args:
        links_df(pandas.DataFrame): links data of network.
        nodes_df(pandas.DataFrame): Nodes data of network.
        traffic_counter_df(pandas.DataFrame): Traffic counters data of network.
        taz_df(pandas.DataFrame): TAZ data.
    
    Returns:
        map_city: Folium map of city with different layers.
    """
    
    #Initializing the folium map
    map_city = add_area_of_interest_layer(nodes_df)

    # Not considering the virtual links as of now in maps plotting
    if not vlinks_flag:
        actual_links_df=links_df.loc[links_df.link_type=="street"]
    else:
        actual_links_df=links_df.copy(deep=True)

    #TAZ & virtual nodes layer
    add_taz_layer(taz_df, map_city)
    add_vnodes_layer(nodes_df, links_df, ref_g_gen, map_city)

    # Actual Nodes layer 
    add_nodes_layer(nodes_df, map_city)

    # Virtual Links layer
    add_virtual_links(links_df, map_city)

    # Actual Links layer
    add_links_layer(actual_links_df, map_city)

    # Links in traversed paths layer
    add_traversed_links_layer(links_df, paths, map_city)

    # Traffic counters layer
    add_traffic_counters_layer(traffic_counter_df, actual_links_df, time, map_city)

    # Baseline Traffic flow layer 
    add_baseline_tf_layer(actual_links_df, traffic_counter_df, map_city)

    # baseline travel time (speed) layer
    add_baseline_abs_speed_layer(actual_links_df, traffic_counter_df, map_city)

    # Toll Links layer
    # Fetching only toll links data.
    actual_toll_links_df = actual_links_df[actual_links_df['toll_price'] != 0]
    #actual_toll_links_df=actual_links_df.loc[actual_links_df.index.isin(df_with_toll.index)]
    add_toll_links_layer(actual_toll_links_df, map_city, time)
 
    # Treated Traffic Flow layer
    add_treated_tf_layer(actual_links_df, folium_map=map_city)

    # Treated travel time layer
    add_treated_abs_speed_layer(actual_links_df, map_city)

    # Changed Traffic flow layer (after intervention)
    add_tf_change_layer(actual_links_df, map_city)

    # changed travel time layer (after intervention)
    add_abs_speed_change_layer(actual_links_df, map_city)

    # Adding the layers control to map
    LayerControl(collapsed=False).add_to(map_city)
    return map_city

def perform_road_closure_visualization(links_df, new_links_df, nodes_df, traffic_counter_df, taz_df,
                                       old_links_key_list, new_links_key_list, closure_links_df,
                                       time, vlinks_flag, paths, ref_g_gen, top_flow_count=3,
                                       top_flow_od_pairs=None, top_flow_paths=None, baseline_flow_paths=None,
                                       alternate_paths_dict=None) :
    
    """To perform road closure visualization in folium map.

    Args:
    
    Returns:
        map_city: Folium map of city with different layers.
    """
    #Initializing the folium map
    #Taking the center of the available data
    #avg_location = nodes_df[['lat', 'lon']].mean()
    #map_city = folium.Map(location=avg_location, zoom_start=13)

    #Initializing the folium map
    map_city = add_area_of_interest_layer(nodes_df)

    # Not considering the virtual links as of now in maps plotting
    if not vlinks_flag:
        actual_links_df=links_df.loc[links_df.link_type=="street"]
    else:
        actual_links_df=links_df.copy(deep=True)

    #TAZ & virtual nodes layer
    add_taz_layer(taz_df, map_city)
    add_vnodes_layer(nodes_df, links_df, ref_g_gen, map_city)

    # Actual Nodes layer 
    add_nodes_layer(nodes_df, map_city)

    # Virtual Links layer
    add_virtual_links(links_df, map_city)

    # Actual Links layer
    add_links_layer(actual_links_df, map_city)

    # Links in traversed paths layer
    add_traversed_links_layer(links_df, paths, map_city)

    # Traffic counters layer
    add_traffic_counters_layer(traffic_counter_df, actual_links_df, time, map_city)

    # Baseline Traffic flow layer 
    add_baseline_tf_layer(actual_links_df, traffic_counter_df, map_city)

    # baseline travel time (speed) layer
    add_baseline_abs_speed_layer(actual_links_df, traffic_counter_df, map_city)

    # Road closure links layer
    add_road_closure_links_layer(closure_links_df,  map_city)

    add_affected_od_old_paths_layer(links_df, closure_links_df, nodes_df, old_links_key_list, map_city)

    add_affected_od_new_paths_layer(links_df, nodes_df, new_links_key_list, map_city)

    # Treated Traffic Flow layer
    add_treated_tf_layer(actual_links_df, new_links_df, folium_map=map_city)

    # Treated travel time layer
    add_treated_abs_speed_layer(actual_links_df, map_city)

    # Changed Traffic flow layer (after intervention)
    add_tf_change_layer(actual_links_df, map_city)

    # Top flow change layer
    add_top_flow_change_layer(
        actual_links_df,
        map_city,
        top_flow_count,
        top_flow_od_pairs,
        top_flow_paths,
        baseline_flow_paths,
        alternate_paths_dict,
    )

    # changed travel time layer (after intervention)
    add_abs_speed_change_layer(actual_links_df, map_city)

    LayerControl(collapsed=False).add_to(map_city)
    
    return map_city

def perform_road_enhancement_visualization(links_df, new_links_df, nodes_df, traffic_counter_df, taz_df,
                                       enhanced_links_df, time, vlinks_flag, paths, ref_g_gen) :
    
    """To perform road enhanced visualization in folium map.

    Args:
    
    Returns:
        map_city: Folium map of city with different layers.
    """
    #Initializing the folium map
    #Taking the center of the available data
    #avg_location = nodes_df[['lat', 'lon']].mean()
    #map_city = folium.Map(location=avg_location, zoom_start=13)

    #Initializing the folium map
    map_city = add_area_of_interest_layer(nodes_df)

    # Not considering the virtual links as of now in maps plotting
    if not vlinks_flag:
        actual_links_df=links_df.loc[links_df.link_type=="street"]

    else:
        actual_links_df=links_df.copy(deep=True)


    #TAZ & virtual nodes layer
    add_taz_layer(taz_df, map_city)
    add_vnodes_layer(nodes_df, links_df, ref_g_gen, map_city)

    # Actual Nodes layer 
    add_nodes_layer(nodes_df, map_city)

    # Virtual Links layer
    add_virtual_links(links_df, map_city)

    # Actual Links layer
    add_links_layer(links_df, map_city)

    # Links in traversed paths layer
    add_traversed_links_layer(links_df, paths, map_city)

    # Traffic counters layer
    add_traffic_counters_layer(traffic_counter_df, actual_links_df, time, map_city)

    # Baseline Traffic flow layer 
    add_baseline_tf_layer(actual_links_df, traffic_counter_df, map_city)

    # baseline travel time (speed) layer
    add_baseline_abs_speed_layer(actual_links_df, traffic_counter_df, map_city)

    # Road enhancement layer
    add_road_enhancement_links_layer(enhanced_links_df,  map_city)

    # Treated Traffic Flow layer
    add_treated_tf_layer(actual_links_df, new_links_df, folium_map=map_city)

    # Treated travel time layer
    add_treated_abs_speed_layer(actual_links_df, map_city)

    # Changed Traffic flow layer (after intervention)
    add_tf_change_layer(actual_links_df, map_city)

    # changed travel time layer (after intervention)
    add_abs_speed_change_layer(actual_links_df, map_city)

    LayerControl(collapsed=False).add_to(map_city)
    
    return map_city