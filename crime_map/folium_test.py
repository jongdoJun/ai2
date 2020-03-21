import pandas as pd
import folium
class FoliumTest:
    def __init__(self):
        pass
    def show_map(self):
        state_geo = './data/us-states.json'
        state_unemployement = './data/us_unemployment.csv'
        state_data = pd.read_csv(state_unemployement)
        m = folium.Map(location=[37, -102], zoom_start=5)
        m.choropleth(
            geo_data = state_geo,
            name = 'choropleth',
            data = state_data,
            columns = ['State', 'Unemployment'],
            key_on = 'feature.id',
            fill_color='YlGn',
            fill_opacity = 0.7,
            line_opacity = 0.2,
            legend_name = 'Unemployement Rate (%)'
        )
        folium.LayerControl().add_to(m)
        m.save('./saved_data/USA.html')