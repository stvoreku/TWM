focal_x = cameraParams.FocalLength(1)
focal_y = cameraParams.FocalLength(2)

principal_x = cameraParams.PrincipalPoint(1)
principal_y = cameraParams.PrincipalPoint(2)

d1 = (focal_x * 1) / 215.2
d2 = (focal_x * 1) / 104.4
d3 = (focal_x * 1) / 70.25

v1 = 100 / (focal_x * 0.01)
v2 = 200 / (focal_x * 0.01)
v3 = 300 / (focal_x * 0.01)