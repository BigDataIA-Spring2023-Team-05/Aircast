use aircast;


SELECT COUNT(*) FROM Zipcode WHERE state_abbr='MA';

DELETE FROM Zipcode;

DELETE FROM Stations;

SELECT COUNT(*) FROM Stations;


SELECT *
FROM Stations
WHERE 6371 * ACOS(COS(RADIANS(42.399501)) * COS(RADIANS(latitude)) * COS(RADIANS(longitude) - RADIANS(-72.201501)) + SIN(RADIANS(42.399501)) * SIN(RADIANS(latitude))) <= 600
ORDER BY 6371 * ACOS(COS(RADIANS(42.399501)) * COS(RADIANS(latitude)) * COS(RADIANS(longitude) - RADIANS(-72.201501)) + SIN(RADIANS(42.399501)) * SIN(RADIANS(latitude))) ASC;


SELECT * FROM Zipcode WHERE zipcode='02119';

SELECT COUNT(*) FROM StationsData WHERE collection_timestamp BETWEEN '2023-04-24T00:00' AND '2023-04-24T23:00';

DELETE FROM StationsData WHERE collection_timestamp BETWEEN '2023-04-24T00:00' AND '2023-04-24T23:00';


SELECT COUNT(*) FROM StationsDataDaily;


DROp TABLE Zipcode;