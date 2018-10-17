% Haversine formula. Used to calculated distance between two points with given geo-coordinates
%
% long1 = longitude in degrees for coordinate 1
% lat1 = latitude in degrees for coordinate 1
% long2 = longitude in degrees for coordinate 1
% lat2 = latitude in degrees for coordinate 1
%
function d = haversine(long1, lat1, long2, lat2)
%

R = 6.371e6;			% earth average radius in meters
lat1 = deg2rad(lat1); 	% convert to radians
lat2 = deg2rad(lat2);
long1 = deg2rad(long1);
long2 = deg2rad(long2);

dlat = lat2 - lat1;
dlong = long2 - long1;

a = (sin(dlat/2)).^2 + cos(lat1).*cos(lat2).*(sin(dlong/2)).^2;
b = sqrt(1-a);
c = 2*atan2(sqrt(a), b);
d = R*c;