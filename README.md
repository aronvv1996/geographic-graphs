# Geographic graphs (working title)

Note: this README file is still a WIP.

## Running the code

Still very much WIP, 'main.py' defines a few methods that can be run but code has to be adjusted to do so. These methods return some figures to a folder names 'figures' in the repository.

## Organization of the code

'data.py' is used to load the datasets.

'graph.py' contains methods to convert loaded data into graphs, and a variety of graph operations.

'clustering.py' contains methods for clustering graphs.

'main.py' contains a handful of methods that make it easy to generate some nice figures.

## Datasets

The project builds on the following datasets:

 - Bordering countries: [country-borders](https://github.com/geodatasource/country-borders/blob/master/GEODATASOURCE-COUNTRY-BORDERS.CSV).
 - Cities: [simplemaps](https://simplemaps.com/data/world-cities), download the free 'Basic' CSV file.
 - Country centroids: [world-countries-centroids](https://github.com/gavinr/world-countries-centroids/blob/master/dist/countries.csv), four entries were incompatible with the other datasets and added manually from [countries_csv](https://developers.google.com/public-data/docs/canonical/countries_csv).
 - Road network: [SEDAC](https://sedac.ciesin.columbia.edu/data/set/groads-global-roads-open-access-v1), requires login. Although the code is by default designed for the Global GeoDataBase, you can download any of the other available GeoDataBases and adjust the 'self.road_file' in the 'data.py' file.
 

 Put the files into a folder called 'data' in the repository.

 ## Report

Click [here](https://www.overleaf.com/read/hccdjstbwvgt) for the LaTeX overleaf file of the report.