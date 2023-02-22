import numpy as np
import matplotlib.pyplot as plt

import math
# p = np.loadtxt('pts.csv',delimiter=',')


#
# x = np.random.rand(20,1)
# y = np.random.rand(20,1)
#
# z = 0.5 * x
#
#
# p = np.zeros((20,3))
# p[:,0] = x[:,0]
# p[:,1] = y[:,0]
# p[:,2] = z[:,0]
# # p = [x,y,z]
#
# p = np.array(p)
# # print(p.shape)
#
# pm = np.mean(p,axis=0)
#
# p2 = np.zeros((20,3))
# p2[:,0] = pm[0]
# p2[:,1] = pm[1]
# p2[:,2] = pm[2]
#
# p = p - p2
# np.savetxt('pts.csv',p,delimiter=',')
p = np.loadtxt('pts.csv',delimiter=',')
titik_pusat = np.mean(p, axis=0)
pt_relatif = p - titik_pusat

u,s,v = np.linalg.svd(pt_relatif)

# v = v.T
# # print(np.dot(v[:,2],p[0,:]))
normal = v[2]
normal = normal / np.linalg.norm(normal)
# pts = list()
# pts = np.array([])
#
# X, Y, Z, U, V, W = pts.reshape(-1,1)

# # print(vektor)
# pT = p.T
# minPlane = int(math.floor(min(min(pT[0]), min(pT[1]), min(pT[2]))))
# maxPlane = int(math.ceil(max(max(pT[0]), max(pT[1]), max(pT[2]))))
# xx, yy = np.meshgrid(range(minPlane,maxPlane), range(minPlane,maxPlane))

# #
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plot the mesh. Each array is 2D, so we flatten them to 1D arrays
ax.plot(p[:,0], p[:,1], p[:,2], 'bo ')
# ax.plot(0 + hasil[0],0 + hasil[1],0 + hasil[2], color='r', linestyle=' ', marker='s')
ax.quiver(titik_pusat[0],titik_pusat[1],titik_pusat[2],normal[0],normal[1], normal[2], linewidths = (5,), edgecolor="red")

plt.show()

# print(hasil)
#
# plt.scatter(p)
# v1 = np.array([1,0,0])
# v2 = np.array([0,1,0])

# print(np.cross(v1,v2))
#
# p1 = np.array([5, 2, 3])
# p2 = np.array([4, 6, 9])
# p3 = np.array([12, 11, 9])
#
# pts = np.array([p1,p2,p3])
# ptsX, ptsY, ptsZ = np.array(pts[:,0]),  np.array(pts[:,1]), np.array(pts[:,2])
#
# # These two vectors are in the plane
# v1 = p3 - p1
# v2 = p2 - p1
#
# # the cross product is a vector normal to the plane
# cp = np.cross(v1, v2)
# a, b, c = cp
#
# # This evaluates a * x3 + b * y3 + c * z3 which equals d
# d = np.dot(cp, p3)
#
# print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))
#
# x = np.linspace(-2, 14, 5)
# y = np.linspace(-2, 14, 5)
# X, Y = np.meshgrid(x, y)
#
# Z = (d - a * X - b * Y) / c
# #
# startX = np.mean(ptsX)
# startY = np.mean(ptsY)
# startZ = np.mean(ptsZ)
# #
# # print(startX, startY, startZ)
# # print(cp)
# normal = cp
# #
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # plot the mesh. Each array is 2D, so we flatten them to 1D arrays
# ax.plot(X.flatten(),
#         Y.flatten(),
#         Z.flatten(), 'bo ')
#
# # plot the original points. We use zip to get 1D lists of x, y and z
# # coordinates.
# # ax.plot(*zip(p1, p2, p3), color='r', linestyle=' ', marker='o')
# ax.plot(p1[0],p1[1],p1[2], color='r', linestyle=' ', marker='s')
# ax.plot(p2[0],p2[1],p2[2], color='r', linestyle=' ', marker='o')
# ax.plot(p3[0],p3[1],p3[2], color='r', linestyle=' ', marker='v')
#
# ax.quiver([startX], [startY], [startZ], [normal[0]], [normal[1]], [normal[2]], linewidths = (5,), edgecolor="red");
# # ax.plot(cp[0],cp[1],cp[2], 'bo')
#
# # adjust the view so we can see the point/plane alignment
# ax.view_init(0, 22)
# plt.tight_layout()
# # plt.savefig('plane.png')
# plt.show()
#
#
