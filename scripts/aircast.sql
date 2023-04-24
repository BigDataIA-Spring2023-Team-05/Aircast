use aircast;


SELECT COUNT(*) FROM Zipcode;

DELETE FROM Zipcode;

DELETE FROM Stations;

SELECT COUNT(*) FROM Stations;


SELECT *
FROM Stations
WHERE 6371 * ACOS(COS(RADIANS(42.324029)) * COS(RADIANS(latitude)) * COS(RADIANS(longitude) - RADIANS(-71.085017)) + SIN(RADIANS(42.324029)) * SIN(RADIANS(latitude))) <= 50
ORDER BY 6371 * ACOS(COS(RADIANS(42.324029)) * COS(RADIANS(latitude)) * COS(RADIANS(longitude) - RADIANS(-71.085017)) + SIN(RADIANS(42.324029)) * SIN(RADIANS(latitude))) ASC;


SELECT * FROM Zipcode WHERE zipcode='02119';