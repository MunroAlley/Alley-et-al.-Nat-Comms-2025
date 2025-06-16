import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

#Munro Alley 2025
#When reading this code remember, I'm a metallurgist, not a coder

#THERMOCHEMICAL DATA
#Activity and Mg3Nd tie-line data from ThermoCalc using the SGTE Alloys and Solutions Database v6.0, which uses the work of Guo et al. 2008
#Standard free energy of formation data from HSC Chemistry 9 in units of J/mol

#Liquidus line for formation of Mg3Nd as f(X_Nd, T); T in degrees C
X_Nd_Mg3Nd_liquidus_data = [0.11135, 0.11486, 0.1196, 0.12336, 0.1234, 0.12845, 0.13786, 0.14774, 0.15805, 0.16872, 0.19364, 0.20651, 0.21959, 0.23284, 0.25]
T_Mg3Nd_liquidus_data = [641.80317, 650.07335, 660.76862, 668.85727, 668.93138, 679.23317, 696.77694, 713.04319, 727.75518, 740.72005, 762.92608, 770.46618, 775.72411, 778.82925, 779.93967]

#Mole fraction Nd corresponding to the activity values stored in thermochemical data dictionary
X_Nd_a_data = [0.000000000001, 0.0234, 0.0484, 0.0734, 0.0984, 0.1234, 0.1484, 0.1734, 0.1984, 0.2234, 0.2484, 0.2734, 0.2984, 0.3234, 0.3484, 0.3734, 0.3984, 0.4234, 0.4484, 0.4734, 0.4984, 0.5234, 0.5484, 0.5734, 0.5984, 0.6234, 0.6484, 0.6734, 0.6984, 0.7234, 0.7484, 0.7734, 0.7984, 0.8234, 0.8484, 0.8734, 0.8984, 0.9234, 0.9484, 0.9734, 0.9984, 1]

#Defining mole fraction Mg from this Nd data, replacing the value that would be zero in the new data with 1E-12 to prevent divide by zero problems
X_Mg_a_data = [1 - value for value in X_Nd_a_data]
X_Mg_a_data[0] = 1
X_Mg_a_data[-1] = 1E-12

#Dictionary for storing thermochemical data at each temperature in C
thermochemical_data = {
    650: {
        'Gf_Nd2O3': -1544928,
        'Gf_MgO': -502361,
        'Gf_NdL': 2321,
        'Gf_NdH2': -64678,
        'a_Nd': np.array([6.48919E-16, 0.00009, 0.00039, 0.00116, 0.00282, 0.00598, 0.01137, 0.01978, 0.03192, 0.04825, 0.06887, 0.09352, 0.12153, 0.15198, 0.18383, 0.21603, 0.24767, 0.27806, 0.30675, 0.33356, 0.35855, 0.38194, 0.40412, 0.42556, 0.44678, 0.46834, 0.49077, 0.5146, 0.54032, 0.56835, 0.59905, 0.63268, 0.66934, 0.709, 0.75134, 0.79576, 0.8413, 0.88652, 0.92949, 0.96776, 0.99838, 1]),
        'a_Mg': np.array([1, 0.96764, 0.91682, 0.85476, 0.78657, 0.71649, 0.64774, 0.58256, 0.52234, 0.46782, 0.41919, 0.37634, 0.33891, 0.30642, 0.27834, 0.25412, 0.23323, 0.21516, 0.19945, 0.18567, 0.17343, 0.16236, 0.15211, 0.1424, 0.13292, 0.12344, 0.11375, 0.1037, 0.09321, 0.0823, 0.07106, 0.05971, 0.04854, 0.0379, 0.02819, 0.01973, 0.01279, 0.00747, 0.00372, 0.00134, 0.00005, 0.000000000000012266])
    },
    675: {
        'Gf_Nd2O3': -1538107,
        'Gf_MgO': -499477,
        'Gf_NdL': 2173,
        'Gf_NdH2': -60714,
        'a_Nd': np.array([8.11279E-16, 0.00011, 0.00047, 0.00136, 0.00324, 0.00675, 0.01266, 0.02175, 0.03471, 0.05194, 0.07351, 0.09906, 0.12789, 0.15903, 0.19141, 0.224, 0.25588, 0.28639, 0.31511, 0.34187, 0.36676, 0.39001, 0.41202, 0.43327, 0.45429, 0.47561, 0.49779, 0.52134, 0.54674, 0.57441, 0.6047, 0.63785, 0.67398, 0.71303, 0.75471, 0.79843, 0.84324, 0.88776, 0.93012, 0.96795, 0.99838, 1]),
        'a_Mg': np.array([1, 0.9679, 0.91784, 0.85684, 0.78983, 0.72092, 0.65323, 0.58894, 0.52943, 0.47542, 0.42715, 0.38452, 0.34718, 0.31471, 0.28657, 0.26226, 0.24124, 0.22303, 0.20717, 0.19324, 0.18083, 0.16959, 0.15918, 0.14927, 0.1396, 0.12989, 0.11995, 0.10961, 0.09878, 0.08747, 0.07579, 0.06393, 0.05221, 0.04098, 0.03067, 0.02162, 0.01413, 0.00833, 0.00419, 0.00153, 0.00006, 1.41928E-14])
    },
    695: {
        'Gf_Nd2O3': -1532655,
        'Gf_MgO': -497169,
        'Gf_NdL': 2046,
        'Gf_NdH2': -57534,
        'a_Nd': np.array([9.61939E-16, 0.00013, 0.00053, 0.00153, 0.0036, 0.00742, 0.01375, 0.02339, 0.037, 0.05496, 0.07726, 0.10351, 0.13296, 0.16462, 0.19741, 0.23027, 0.26233, 0.29292, 0.32164, 0.34835, 0.37315, 0.39628, 0.41816, 0.43925, 0.46009, 0.48123, 0.50321, 0.52654, 0.55169, 0.57907, 0.60904, 0.64182, 0.67754, 0.71612, 0.75729, 0.80047, 0.84472, 0.88871, 0.9306, 0.96809, 0.99838, 1]),
        'a_Mg': np.array([1, 0.9681, 0.91862, 0.85842, 0.79233, 0.72432, 0.65744, 0.59385, 0.5349, 0.48131, 0.43333, 0.39087, 0.35363, 0.32118, 0.29302, 0.26864, 0.24754, 0.22923, 0.21326, 0.19921, 0.18669, 0.17532, 0.16478, 0.15474, 0.14492, 0.13504, 0.1249, 0.11434, 0.10325, 0.09164, 0.07961, 0.06735, 0.05519, 0.0435, 0.0327, 0.02318, 0.01524, 0.00905, 0.00459, 0.00169, 0.00007, 1.58615E-14])
    },
    700: {
        'Gf_Nd2O3': -1531294,
        'Gf_MgO': -496592,
        'Gf_NdL': 2014,
        'Gf_NdH2': -56738,
        'a_Nd': np.array([1.00269E-15, 0.00013, 0.00055, 0.00158, 0.0037, 0.00759, 0.01403, 0.02381, 0.03758, 0.05572, 0.0782, 0.10462, 0.13422, 0.16601, 0.19889, 0.23183, 0.26392, 0.29453, 0.32325, 0.34995, 0.37472, 0.39782, 0.41966, 0.44072, 0.46152, 0.48261, 0.50454, 0.52781, 0.5529, 0.58022, 0.6101, 0.6428, 0.67841, 0.71688, 0.75792, 0.80096, 0.84508, 0.88894, 0.93072, 0.96812, 0.99838, 1]),
        'a_Mg': np.array([1, 0.96815, 0.91881, 0.85881, 0.79294, 0.72515, 0.65847, 0.59506, 0.53624, 0.48276, 0.43485, 0.39244, 0.35522, 0.32278, 0.29461, 0.27022, 0.2491, 0.23077, 0.21477, 0.20069, 0.18814, 0.17675, 0.16618, 0.1561, 0.14624, 0.13633, 0.12614, 0.11552, 0.10437, 0.09268, 0.08056, 0.06821, 0.05594, 0.04414, 0.03322, 0.02358, 0.01552, 0.00923, 0.00469, 0.00174, 0.00007, 1.62963E-14])
    },
    725: {
        'Gf_Nd2O3': -1524488,
        'Gf_MgO': -493707,
        'Gf_NdL': 1846,
        'Gf_NdH2': -52752,
        'a_Nd': np.array([1.22618E-15, 0.00016, 0.00065, 0.00182, 0.00419, 0.00847, 0.01546, 0.02594, 0.04053, 0.05955, 0.08293, 0.11019, 0.14054, 0.17293, 0.20627, 0.23952, 0.27179, 0.30247, 0.33118, 0.3578, 0.38244, 0.40539, 0.42705, 0.44791, 0.4685, 0.48936, 0.51104, 0.53404, 0.55882, 0.58579, 0.61528, 0.64753, 0.68264, 0.72055, 0.76099, 0.80338, 0.84684, 0.89006, 0.93129, 0.96829, 0.99838, 1]),
        'a_Mg': np.array([1, 0.96839, 0.91973, 0.86069, 0.7959, 0.72918, 0.6635, 0.60093, 0.54279, 0.48982, 0.44228, 0.40011, 0.36303, 0.33063, 0.30246, 0.27801, 0.2568, 0.23836, 0.22225, 0.20804, 0.19536, 0.18383, 0.17311, 0.16288, 0.15285, 0.14274, 0.13232, 0.12143, 0.10997, 0.09792, 0.08538, 0.07255, 0.05974, 0.04736, 0.03584, 0.0256, 0.01698, 0.01018, 0.00522, 0.00195, 0.00008, 1.85854E-14])
    },
    750: {
        'Gf_Nd2O3': -1517689,
        'Gf_MgO': -490821,
        'Gf_NdL': 1670,
        'Gf_NdH2': -48755,
        'a_Nd': np.array([1.48482E-15, 0.00019, 0.00075, 0.00208, 0.00472, 0.00941, 0.01696, 0.02814, 0.04355, 0.06344, 0.08769, 0.11577, 0.14681, 0.17978, 0.21354, 0.24706, 0.2795, 0.31022, 0.3389, 0.36543, 0.38994, 0.41272, 0.43419, 0.45485, 0.47523, 0.49587, 0.5173, 0.54002, 0.5645, 0.59113, 0.62025, 0.65207, 0.68669, 0.72407, 0.76391, 0.80569, 0.84851, 0.89113, 0.93183, 0.96845, 0.99838, 1]),
        'a_Mg': np.array([1, 0.96862, 0.92061, 0.86248, 0.79872, 0.73305, 0.66831, 0.60656, 0.5491, 0.49664, 0.44948, 0.40755, 0.37061, 0.33828, 0.31011, 0.28562, 0.26435, 0.24582, 0.2296, 0.21528, 0.20248, 0.19083, 0.17998, 0.1696, 0.1594, 0.14911, 0.13848, 0.12734, 0.11558, 0.10318, 0.09022, 0.07692, 0.0636, 0.05064, 0.03852, 0.02768, 0.01849, 0.01117, 0.00578, 0.00219, 0.00009, 2.10589E-14])
    },
    775: {
        'Gf_Nd2O3': -1510895,
        'Gf_MgO': -487934,
        'Gf_NdL': 1488,
        'Gf_NdH2': -44748,
        'a_Nd': np.array([1.78167E-15, 0.00022, 0.00087, 0.00236, 0.00529, 0.0104, 0.01852, 0.03042, 0.04663, 0.06738, 0.09248, 0.12133, 0.15305, 0.18655, 0.2207, 0.25447, 0.28703, 0.31779, 0.34642, 0.37284, 0.39721, 0.41982, 0.44111, 0.46157, 0.48173, 0.50214, 0.52333, 0.54579, 0.56997, 0.59627, 0.62501, 0.65641, 0.69057, 0.72743, 0.76671, 0.80789, 0.85011, 0.89215, 0.93235, 0.9686, 0.99838, 1]),
        'a_Mg': np.array([1, 0.96883, 0.92144, 0.86419, 0.80142, 0.73674, 0.67293, 0.61198, 0.55517, 0.50323, 0.45643, 0.41476, 0.37798, 0.34573, 0.31758, 0.29307, 0.27174, 0.25313, 0.23683, 0.22241, 0.20951, 0.19774, 0.18677, 0.17626, 0.16591, 0.15544, 0.14461, 0.13323, 0.12118, 0.10844, 0.09509, 0.08134, 0.06749, 0.05398, 0.04127, 0.02983, 0.02005, 0.01221, 0.00638, 0.00243, 0.0001, 2.37212E-14])
    },
    800: {
        'Gf_Nd2O3': -1504107,
        'Gf_MgO': -485047,
        'Gf_NdL': 1300,
        'Gf_NdH2': -40729,
        'a_Nd': np.array([2.11978E-15, 0.00026, 0.001, 0.00267, 0.00589, 0.01144, 0.02014, 0.03275, 0.04978, 0.07137, 0.09729, 0.12689, 0.15925, 0.19324, 0.22775, 0.26175, 0.29441, 0.32518, 0.35374, 0.38005, 0.40427, 0.42671, 0.44781, 0.46806, 0.48801, 0.5082, 0.52914, 0.55134, 0.57523, 0.60121, 0.62959, 0.66059, 0.69429, 0.73065, 0.76939, 0.80999, 0.85164, 0.89312, 0.93284, 0.96875, 0.99838, 1]), #Must be in same dimension and order as X_Nd, same for a_Mg
        'a_Mg': np.array([1, 0.96904, 0.92224, 0.86583, 0.80401, 0.74028, 0.67736, 0.61719, 0.56102, 0.50958, 0.46317, 0.42176, 0.38514, 0.35299, 0.32487, 0.30035, 0.27898, 0.26031, 0.24393, 0.22943, 0.21643, 0.20456, 0.19348, 0.18285, 0.17236, 0.16173, 0.15071, 0.1391, 0.12678, 0.11371, 0.09998, 0.08578, 0.07143, 0.05737, 0.04407, 0.03202, 0.02167, 0.01329, 0.007, 0.0027, 0.00012, 2.65672E-14])
    },
    825: {
        'Gf_Nd2O3': -1497323,
        'Gf_MgO': -482160,
        'Gf_NdL': 1108,
        'Gf_NdH2': -36701,
        'a_Nd': np.array([2.50219E-15, 0.0003, 0.00114, 0.003, 0.00653, 0.01253, 0.02183, 0.03515, 0.05297, 0.0754, 0.10211, 0.13244, 0.16539, 0.19986, 0.23469, 0.26888, 0.30163, 0.33239, 0.36088, 0.38706, 0.41113, 0.43339, 0.45429, 0.47435, 0.49409, 0.51405, 0.53475, 0.55669, 0.5803, 0.60597, 0.63399, 0.6646, 0.69786, 0.73373, 0.77195, 0.81201, 0.85309, 0.89405, 0.93331, 0.96889, 0.99838, 1]), #Must be in same dimension and order as X_Nd, same for a_Mg
        'a_Mg': np.array([1, 0.96924, 0.923, 0.86739, 0.80648, 0.74368, 0.68162, 0.6222, 0.56667, 0.51573, 0.46968, 0.42854, 0.39211, 0.36005, 0.33198, 0.30747, 0.28607, 0.26735, 0.25091, 0.23634, 0.22325, 0.21129, 0.20011, 0.18937, 0.17875, 0.16797, 0.15677, 0.14494, 0.13236, 0.11898, 0.10488, 0.09025, 0.07541, 0.0608, 0.04691, 0.03427, 0.02333, 0.0144, 0.00764, 0.00297, 0.00013, 2.96051E-14])
    },
    850: {
        'Gf_Nd2O3': -1490545,
        'Gf_MgO': -479272,
        'Gf_NdL': 912,
        'Gf_NdH2': -32662,
        'a_Nd': np.array([2.93186E-15, 0.00034, 0.00129, 0.00335, 0.0072, 0.01366, 0.02357, 0.03761, 0.05622, 0.07945, 0.10695, 0.13796, 0.17149, 0.20639, 0.24152, 0.27588, 0.30869, 0.33943, 0.36783, 0.39389, 0.41778, 0.43987, 0.46058, 0.48043, 0.49996, 0.5197, 0.54017, 0.56186, 0.58519, 0.61055, 0.63823, 0.66845, 0.70129, 0.73669, 0.77441, 0.81394, 0.85449, 0.89494, 0.93376, 0.96902, 0.99838, 1]), #Must be in same dimension and order as X_Nd, same for a_Mg
        'a_Mg': np.array([1, 0.96942, 0.92373, 0.86888, 0.80885, 0.74694, 0.68571, 0.62703, 0.57211, 0.52167, 0.476, 0.43513, 0.39888, 0.36694, 0.33892, 0.31442, 0.29301, 0.27426, 0.25776, 0.24313, 0.22997, 0.21793, 0.20666, 0.19581, 0.18508, 0.17416, 0.16279, 0.15076, 0.13792, 0.12425, 0.10978, 0.09473, 0.07941, 0.06427, 0.0498, 0.03656, 0.02503, 0.01556, 0.00832, 0.00326, 0.00014, 0.000000000000032837])
    }
}

#Activity Interpolation Calculations
X_Nd_smooth = np.linspace(0, 1, 201)
X_Mg_smooth = [1 - value for value in X_Nd_smooth]

for T in thermochemical_data:
    #Calculating gamma
    gamma_Nd = thermochemical_data[T]['a_Nd'] / X_Nd_a_data
    gamma_Mg = thermochemical_data[T]['a_Mg'] / X_Mg_a_data
    
    #Appending to thermochemical data
    thermochemical_data[T]['gamma_Nd'] = gamma_Nd
    thermochemical_data[T]['gamma_Mg'] = gamma_Mg
    
    #Interpolation, defining gamma Mg in terms of X Nd
    gamma_Nd_interp_func = interp1d(X_Nd_a_data, thermochemical_data[T]['gamma_Nd'], kind='linear', fill_value="extrapolate")
    gamma_Mg_interp_func = interp1d(X_Nd_a_data, thermochemical_data[T]['gamma_Mg'], kind='linear', fill_value="extrapolate")

    #Making smooth interpolated plots of a and gamma
    gamma_Nd_smooth = gamma_Nd_interp_func(X_Nd_smooth)
    gamma_Mg_smooth = gamma_Mg_interp_func(X_Nd_smooth)
    a_Nd_smooth = X_Nd_smooth * gamma_Nd_smooth
    a_Mg_smooth = X_Mg_smooth * gamma_Mg_smooth
    thermochemical_data[T]['gamma_Mg_smooth'] = gamma_Mg_smooth
    thermochemical_data[T]['gamma_Nd_smooth'] = gamma_Nd_smooth
    thermochemical_data[T]['a_Mg_smooth'] = a_Mg_smooth
    thermochemical_data[T]['a_Nd_smooth'] = a_Nd_smooth
    
    #Plotting the interpolated activity
    plt.plot(X_Nd_smooth, a_Nd_smooth, '-', label='Interpolated Nd Activity')
    plt.plot(X_Nd_smooth, a_Mg_smooth, '-', label='Interpolated Mg Activity')
    plt.plot(X_Nd_a_data, thermochemical_data[T]['a_Nd'], 'o', label='Nd Activity Data')
    plt.plot(X_Nd_a_data, thermochemical_data[T]['a_Mg'], 'o', label='Mg Activity Data')
    plt.xlabel('Mole Fraction Nd')
    plt.ylabel('Activity')
    plt.title(f'Interpolation of Activity at {T}')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.show()
    
    #Plotting the interpolated activity coefficients
    plt.plot(X_Nd_smooth, gamma_Nd_smooth, '-', label='Nd Activity')
    plt.plot(X_Nd_smooth, gamma_Mg_smooth, '-', label='Mg Activity')
    plt.plot(X_Nd_a_data, thermochemical_data[T]['gamma_Nd'], 'o', label='Nd Activity Data')
    plt.plot(X_Nd_a_data, thermochemical_data[T]['gamma_Mg'], 'o', label='Mg Activity Data')
    plt.xlabel('Mole Fraction Nd')
    plt.ylabel('Activity Coefficient')
    plt.title(f'Interpolation of Activity Coefficient at {T}')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.show()

#Interpolation of Mg3Nd liquidus
Mg3Nd_interp_func = interp1d(T_Mg3Nd_liquidus_data, X_Nd_Mg3Nd_liquidus_data, kind='quadratic', fill_value = "extrapolate") #Interpolating with degree 2 polynomial, return X_Nd as f(T)
T_Mg3Nd_smooth = np.linspace(640, 779.93967, 100)
X_Nd_Mg3Nd_smooth = Mg3Nd_interp_func(T_Mg3Nd_smooth)

#Calculating for each T
for T in thermochemical_data:
    thermochemical_data[T]['X_Nd_Mg3Nd'] = Mg3Nd_interp_func(T)

# Plotting the interpolated activity
plt.plot(X_Nd_Mg3Nd_smooth, T_Mg3Nd_smooth, '-', label='Interpolated')
plt.plot(X_Nd_Mg3Nd_liquidus_data, T_Mg3Nd_liquidus_data, 'o', label='Data')
plt.xlabel('Mole Fraction Nd')
plt.ylabel('T (C)')
plt.title('Interpolation of Mg3Nd Liquidus')
plt.xlim(0, 0.3)
plt.ylim(650, 850)
plt.legend()
plt.grid()
plt.show()

#REDUCTION CALCULATIONS
#Start by defining non-standard delta G value for reduction of Nd2O3 by Mg
def calc_delta_G(X_Nd, is_ideal, Gf_MgO, Gf_Nd2O3, Gf_NdL, T):
    if is_ideal == 'true':
        a_Nd = X_Nd
        a_Mg = 1 - X_Nd
    else:
        a_Nd = X_Nd * gamma_Nd_interp_func(X_Nd)
        a_Mg = (1 - X_Nd) * gamma_Mg_interp_func(X_Nd)
        
    delta_G = (3 * Gf_MgO + 2 * Gf_NdL - Gf_Nd2O3) + 8.314 * (T + 273.15) * np.log(a_Nd**2 / a_Mg**3)
    
    return(delta_G)

#Function to calculate the equilibrium reduction composition
def calc_eq_comp(is_ideal, Gf_MgO, Gf_Nd2O3, Gf_NdL, T, iterations, increment, precision):
    X_Nd_guess = increment
    
    for i in range(iterations):
        
        delta_G_guess = calc_delta_G(X_Nd_guess, is_ideal, Gf_MgO, Gf_Nd2O3, Gf_NdL, T)
    
        if delta_G_guess < 0:
            X_Nd_guess += increment
        elif delta_G_guess > precision:
            X_Nd_guess -= increment
            increment = increment / 10
        else:
            eq_comp = X_Nd_guess
            
        if i == iterations:
            print('Not enough iterations to converge')
            
    return eq_comp

#Find equilibrium reduction composition for both ideal and non-ideal Mg-Nd solution
for T in thermochemical_data:
    Gf_MgO = thermochemical_data[T]['Gf_MgO']
    Gf_Nd2O3 = thermochemical_data[T]['Gf_Nd2O3']
    Gf_NdL = thermochemical_data[T]['Gf_NdL']
    is_ideal = 'true'
    precision = 0.000001
    increment = 0.001
    iterations = 10000
    
    thermochemical_data[T]['X_Nd_eq_ideal'] = calc_eq_comp(is_ideal, Gf_MgO, Gf_Nd2O3, Gf_NdL, T, iterations, increment, precision)
    
    is_ideal = 'false'
    precision = 0.000001
    increment = 0.001
    iterations = 10000
    
    thermochemical_data[T]['X_Nd_eq_nonideal'] = calc_eq_comp(is_ideal, Gf_MgO, Gf_Nd2O3, Gf_NdL, T, iterations, increment, precision)

#Plotting equilibrium composition as a function of T
eq_comp_list_ideal = [values['X_Nd_eq_ideal'] for values in thermochemical_data.values()]
eq_comp_list_nonideal = [values['X_Nd_eq_nonideal'] for values in thermochemical_data.values()]

temps = [T for T in thermochemical_data]

plt.plot(temps, eq_comp_list_ideal, '-', label='Ideal Solution')
plt.plot(temps, eq_comp_list_nonideal, '-', label='Non-Ideal Solution')
plt.plot(T_Mg3Nd_smooth, X_Nd_Mg3Nd_smooth, '-', label='Mg3Nd Forms')
plt.xlabel('T (C)')
plt.ylabel('Equilibrium X Nd')
plt.title('Equilibrium Composition vs. Temperature')
plt.xlim(650, 850)
plt.ylim(0, 0.25)
plt.legend()
plt.grid()
plt.show()

#Calculate delta G as a function of X Nd
X_Nd_range = np.linspace(0.000001, 0.999999, 1000)

for T in thermochemical_data:
    is_ideal = 'true'
    Gf_MgO = thermochemical_data[T]['Gf_MgO']
    Gf_Nd2O3 = thermochemical_data[T]['Gf_Nd2O3']
    Gf_NdL = thermochemical_data[T]['Gf_NdL']
    
    delta_G_ns = calc_delta_G(X_Nd_range, is_ideal, Gf_MgO, Gf_Nd2O3, Gf_NdL, T)
    thermochemical_data[T]['delta_G_ns_ideal'] = delta_G_ns
    
    is_ideal = 'false'
    delta_G_ns = calc_delta_G(X_Nd_range, is_ideal, Gf_MgO, Gf_Nd2O3, Gf_NdL, T)
    thermochemical_data[T]['delta_G_ns_nonideal'] = delta_G_ns
    
    plt.plot(X_Nd_range, thermochemical_data[T]['delta_G_ns_ideal'], '-', label='Ideal Solution')
    plt.plot(X_Nd_range, thermochemical_data[T]['delta_G_ns_nonideal'], '-', label='Non-Ideal Solution')
    plt.plot(T_Mg3Nd_smooth, X_Nd_Mg3Nd_smooth, '-', label='Mg3Nd Forms')
    plt.xlabel('X Nd')
    plt.ylabel('Delta G (J/mol of Nd2O3)')
    plt.title(f'Free Energy vs. Composition at {T}')
    plt.xlim(0, 1)
    plt.ylim(-250000, 250000)
    plt.legend()
    plt.grid()
    plt.show()

#Initial Reactant Ratio Calculation
Nd2O3_i = np.linspace(0, 1, 1001)
Mg_i = 3
X_Nd_full_red = (2 * Nd2O3_i) / ((2 * Nd2O3_i) + (Mg_i - (3 * Nd2O3_i)))

for T in thermochemical_data:
    plt.plot(Nd2O3_i, X_Nd_full_red, '-', label='X Nd at Full Reduction')
    plt.xlabel('Fraction of Stoichiometric Ratio')
    plt.axhline(thermochemical_data[T]['X_Nd_eq_ideal'], color='green', label="Eq. X Nd, Ideal")
    plt.axhline(thermochemical_data[T]['X_Nd_eq_nonideal'], color='orange', label="Eq. X Nd, Non-Ideal")
    plt.axhline(thermochemical_data[T]['X_Nd_Mg3Nd'], color='red', label="Mg3Nd Forms")
    plt.ylabel('X Nd')
    plt.title(f'X Nd vs. Fraction of Stoichiometric Ratio at {T}')
    plt.xlim(0, 0.4)
    plt.ylim(0, 0.3)
    plt.legend()
    plt.grid()
    plt.show()
    
#HYDRIDE PRECIPITATION CALCULATIONS
#Start by defining non-standard delta G value for hydride precipitation
def calc_hyd_delta_G(X_Nd, is_ideal, Gf_NdH2, Gf_NdL, T):
    if is_ideal == 'true':
        a_Nd = X_Nd
    else:
        a_Nd = X_Nd * gamma_Nd_interp_func(X_Nd)
        
    delta_G = Gf_NdH2 - Gf_NdL + (8.314 * (T + 273.15) * np.log(1 / (a_Nd)))
    
    return(delta_G)

#Calculate G for 695 C
Gf_NdH2 = thermochemical_data[695]['Gf_NdH2']
Gf_NdL = thermochemical_data[695]['Gf_NdL']
is_ideal = 'true'
T = 695
hyd_delta_G_ns = calc_hyd_delta_G(X_Nd_range, is_ideal, Gf_NdH2, Gf_NdL, T)
thermochemical_data[695]['hyd_delta_G_ns_ideal'] = hyd_delta_G_ns

is_ideal = 'false'
T = 695
hyd_delta_G_ns = calc_hyd_delta_G(X_Nd_range, is_ideal, Gf_NdH2, Gf_NdL, T)
thermochemical_data[695]['hyd_delta_G_ns_nonideal'] = hyd_delta_G_ns

plt.plot(X_Nd_range, thermochemical_data[695]['hyd_delta_G_ns_ideal'], '-', label='Ideal Solution')
plt.plot(X_Nd_range, thermochemical_data[695]['hyd_delta_G_ns_nonideal'], '-', label='Non-Ideal Solution')
plt.plot(T_Mg3Nd_smooth, X_Nd_Mg3Nd_smooth, '-', label='Mg3Nd Forms')
plt.xlabel('X Nd')
plt.ylabel('Delta G (J/mol of Nd2O3)')
plt.title(f'Free Energy vs. Composition at 695')
plt.xlim(0, 1)
plt.ylim(-250000, 250000)
plt.legend()
plt.grid()
plt.show()


#Function for determining equilibrium hydriding X Nd
def calc_hyd_eq_comp(is_ideal, Gf_NdH2, Gf_NdL, iterations, increment, precision, T, P_H2):
    a_Nd = (np.exp((Gf_NdH2 - Gf_NdL) / (8.314 * (T + 273.15)))) / P_H2
    X_Nd_guess = precision
    
    if is_ideal == 'true':
        hyd_eq_comp = a_Nd
    else:
        if a_Nd > 1:
            hyd_eq_comp = a_Nd
        else:
            for i in range(iterations):
                if i == iterations:
                    print('Not enough iterations to converge')
                a_Nd_guess = X_Nd_guess * gamma_Nd_interp_func(X_Nd_guess)
                a_Nd_dif = a_Nd_guess - a_Nd
                if a_Nd_dif < 0:
                    X_Nd_guess += increment
                elif a_Nd_dif > precision:
                    X_Nd_guess -= increment
                    increment = increment / 10
                else:
                    hyd_eq_comp = X_Nd_guess
                    break
        
    return(hyd_eq_comp)
    
#Calculating Eq. X Nd as a function of PH2
P_H2_range = np.linspace(0.000001, 1.1, 1100)

for T in thermochemical_data:
    
    X_Nd_eq_hyd_ideal = []
    X_Nd_eq_hyd_nonideal = []
    
    for P_H2 in P_H2_range:
        Gf_NdH2 = thermochemical_data[T]['Gf_NdH2']
        Gf_NdL = thermochemical_data[T]['Gf_NdL']
        is_ideal = 'true'
        precision = 0.0000001
        increment = 0.001
        iterations = 10000
        X_Nd_eq_hyd_ideal_val = calc_hyd_eq_comp(is_ideal, Gf_NdH2, Gf_NdL, iterations, increment, precision, T, P_H2)
        X_Nd_eq_hyd_ideal.append(X_Nd_eq_hyd_ideal_val)
        
        
        is_ideal = 'false'
        precision = 0.0000001
        increment = 0.001
        iterations = 10000
        X_Nd_eq_hyd_nonideal_val = calc_hyd_eq_comp(is_ideal, Gf_NdH2, Gf_NdL, iterations, increment, precision, T, P_H2)
        X_Nd_eq_hyd_nonideal.append(X_Nd_eq_hyd_nonideal_val)
        
    thermochemical_data[T]['X_Nd_eq_hyd_ideal'] = X_Nd_eq_hyd_ideal
    thermochemical_data[T]['X_Nd_eq_hyd_nonideal'] = X_Nd_eq_hyd_nonideal
    
    #Plotting results of hydride precipitation calculation
    plt.plot(P_H2_range,  thermochemical_data[T]['X_Nd_eq_hyd_ideal'], '-', label='Ideal Solution')
    plt.plot(P_H2_range, thermochemical_data[T]['X_Nd_eq_hyd_nonideal'], '-', label='Non-Ideal Solution')
    plt.xlabel('P H2')
    plt.ylabel('Equilibrium X Nd')
    plt.axhline(thermochemical_data[T]['X_Nd_Mg3Nd'], color='red', label="Mg3Nd Forms")
    plt.title(f'Equilibrium Composition vs. H2 Pressure at {T}')
    plt.xlim(0, 1.1)
    plt.ylim(0, 0.1)
    plt.legend()
    plt.grid()
    plt.show()

print('Calculations complete')

#Create DataFrame and export data to Excel
"""
data = {
    #Activity Data
    'X Nd Smooth for a and gamma': X_Nd_smooth,
    'Gamma Mg Smooth at 695': thermochemical_data[695]['gamma_Mg_smooth'],
    'Gamma Mg Smooth at 850': thermochemical_data[850]['gamma_Mg_smooth'],
    'Activity Mg Smooth at 695': thermochemical_data[695]['a_Mg_smooth'],
    'Activity Mg Smooth at 850': thermochemical_data[850]['a_Mg_smooth'],
    'Gamma Nd Smooth at 695': thermochemical_data[695]['gamma_Nd_smooth'],
    'Gamma Nd Smooth at 850': thermochemical_data[850]['gamma_Nd_smooth'],
    'Activity Nd Smooth at 695': thermochemical_data[695]['a_Nd_smooth'],
    'Activity Nd Smooth at 850': thermochemical_data[850]['a_Nd_smooth'],
    
    #Reduction Data
    #Non-standard free energy
    'X Nd for Delta G': X_Nd_range,
    'Delta G Ideal 850': thermochemical_data[850]['delta_G_ns_ideal'],
    'Delta G Non-Ideal 850': thermochemical_data[850]['delta_G_ns_nonideal'],
    
    #Reduction vs Initial Reactant Ratio
    'Initial Fraction of Stoichiometric Ratio': Nd2O3_i,
    'X Nd at Full reduction': X_Nd_full_red,
    'Reduction X Nd Eq Ideal': thermochemical_data[850]['X_Nd_eq_ideal'],
    'Reduction X Nd Eq Non-Ideal': thermochemical_data[850]['X_Nd_eq_nonideal'],
    
    #Equilibrium Reduction Concentration as a function of temperature
    'Temperature (C)': temps,
    'Ideal Eq Concentration': eq_comp_list_ideal,
    'Non-Ideal Eq Concentration': eq_comp_list_nonideal,
    
    #Hydride Precipitation Data
    'PH2': P_H2_range,
    "Hydride Precipitation X Nd Eq Ideal 650": thermochemical_data[650]['X_Nd_eq_hyd_ideal'],
    "Hydride Precipitation X Nd Eq Non-Ideal 650": thermochemical_data[650]['X_Nd_eq_hyd_nonideal'],
    "Hydride Precipitation X Nd Eq Ideal 675": thermochemical_data[675]['X_Nd_eq_hyd_ideal'],
    "Hydride Precipitation X Nd Eq Non-Ideal 675": thermochemical_data[675]['X_Nd_eq_hyd_nonideal'],
    "Hydride Precipitation X Nd Eq Ideal 700": thermochemical_data[700]['X_Nd_eq_hyd_ideal'],
    "Hydride Precipitation X Nd Eq Non-Ideal 700": thermochemical_data[700]['X_Nd_eq_hyd_nonideal'],
    "Hydride Precipitation X Nd Eq Ideal 725": thermochemical_data[725]['X_Nd_eq_hyd_ideal'],
    "Hydride Precipitation X Nd Eq Non-Ideal 725": thermochemical_data[725]['X_Nd_eq_hyd_nonideal'],
    "Hydride Precipitation X Nd Eq Ideal 750": thermochemical_data[750]['X_Nd_eq_hyd_ideal'],
    "Hydride Precipitation X Nd Eq Non-Ideal 750": thermochemical_data[750]['X_Nd_eq_hyd_nonideal'],
    "Hydride Precipitation X Nd Eq Ideal 775": thermochemical_data[775]['X_Nd_eq_hyd_ideal'],
    "Hydride Precipitation X Nd Eq Non-Ideal 775": thermochemical_data[775]['X_Nd_eq_hyd_nonideal'],
    "Hydride Precipitation X Nd Eq Ideal 800": thermochemical_data[800]['X_Nd_eq_hyd_ideal'],
    "Hydride Precipitation X Nd Eq Non-Ideal 800": thermochemical_data[800]['X_Nd_eq_hyd_nonideal'],
    "Hydride Precipitation X Nd Eq Ideal 825": thermochemical_data[825]['X_Nd_eq_hyd_ideal'],
    "Hydride Precipitation X Nd Eq Non-Ideal 825": thermochemical_data[825]['X_Nd_eq_hyd_nonideal'],
    "Hydride Precipitation X Nd Eq Ideal 850": thermochemical_data[850]['X_Nd_eq_hyd_ideal'],
    "Hydride Precipitation X Nd Eq Non-Ideal 850": thermochemical_data[850]['X_Nd_eq_hyd_nonideal'],
    
    'X Nd Range': X_Nd_range,
    'Delta G Ideal': thermochemical_data[695]['hyd_delta_G_ns_ideal'],
    'Delta G Non-Ideal': thermochemical_data[695]['hyd_delta_G_ns_nonideal']
}

#Check the maximum length of lists (considering arrays as well)
max_length = max(len(v) if isinstance(v, (list, np.ndarray)) else 1 for v in data.values())

#Adjust lengths of all lists/arrays
for key in data:
    if isinstance(data[key], np.ndarray):  # If the data is a numpy array
        data[key] = data[key].tolist()  # Convert to list for consistency
    if isinstance(data[key], (list, np.ndarray)):  # If the data is a list or array
        while len(data[key]) < max_length:
            data[key].append(None)  # Or another placeholder (e.g., np.nan)
    else:  # If the data is a scalar (e.g., float or int)
        data[key] = [data[key]] * max_length  # Expand scalar to match max_length

df = pd.DataFrame(data)
output_file = 'Mg_Nd_Calculation_Output.xlsx'
df.to_excel(output_file, index=False)
print(f'Data has been written to {output_file}.')
"""