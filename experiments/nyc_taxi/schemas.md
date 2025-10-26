Required Columns
----------------

The loader expects the latest NYC TLC schema with the following columns
(case-insensitive checks are performed):

- Pickup datetime: one of
  - `tpep_pickup_datetime`
  - `lpep_pickup_datetime`
  - `pickup_datetime`
- Dropoff datetime: one of
  - `tpep_dropoff_datetime`
  - `lpep_dropoff_datetime`
  - `dropoff_datetime`
- Pickup longitude: one of
  - `pickup_longitude`
  - `pickup_lon`
  - `pickup_long`
- Pickup latitude: one of
  - `pickup_latitude`
  - `pickup_lat`
- Dropoff longitude: one of
  - `dropoff_longitude`
  - `dropoff_lon`
  - `dropoff_long`
- Dropoff latitude: one of
  - `dropoff_latitude`
  - `dropoff_lat`

The loader reports a clear error if any required columns are missing.

Coordinate Sanity Bounds
------------------------

By default we keep trips with coordinates in:
- Latitude: [40.0, 41.0]
- Longitude: [-75.0, -73.0]

Adjust this if your dataset covers a different bounding box.
