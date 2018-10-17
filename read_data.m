% read_data.m
%
% Reads file with one headerline. File is delimited by commas. Format string needs to be adapted to specific file contents.
%
% Format below is adapted to read file train_epoch.csv
%


[nr, t_id, vendor, pickup_td, dropoff_td, count, pickup_long, pickup_lat, dropoff_long, dropoff_lat, flag, ...
	trip_duration, pickup_t, dropoff_t] = textread("train_epoch.csv", "%f %s %s %s %s %f %f %f %f %f %s %f %f %f", 'delimiter' , ',', 'headerlines', 1);