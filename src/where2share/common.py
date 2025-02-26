import pyproj

wgs84 = pyproj.CRS.from_epsg(4326)
web_mercator = pyproj.CRS.from_epsg(3857)
utm_32N = pyproj.CRS.from_epsg(25832)
