# -*- coding: utf-8 -*-
""" Tests for the AV_pairing module
Created on Sat Feb 24 21:27:33 2018

@author: tbeleyur
"""

import unittest
import numpy as np
import scipy.spatial as spatial
import pandas as pd

from AV_pairing import *

class TestReliablePoints(unittest.TestCase):
    '''
    '''

    def test_simpledatasets(self):

        np.random.seed(111)

        num_tespts = 20
        a_points = np.random.normal(0,1,num_tespts*3).reshape((-1,3))

        # translate in x and y points
        b_points = np.copy(a_points) + np.random.normal(0,0.005,
                                            num_tespts*3).reshape((-1,3))

        # create non-conformal points in b
        num_nonconfpts = 5
        b_points[:num_nonconfpts,0] += np.random.normal(10,20,num_nonconfpts)

        expected_reliablepts = np.arange(num_nonconfpts,b_points.shape[0])
        obtained_reliablepts , dists = choose_reliable_points(a_points,
                                                              b_points, 0.1 )


        same_reliablepts = np.array_equal(expected_reliablepts,
                                                          obtained_reliablepts)

        self.assertTrue(same_reliablepts)

class TestRigidTransform(unittest.TestCase):
    '''TO DO : write the test !!
    '''


class TestAssignCallstoTrajs(unittest.TestCase):
    '''
    '''

    def test_asssigncallstotrajs(self):

        #create a video based trajectory :
        fps = 25
        x = np.linspace(0,1,fps)
        y = np.sin(2*np.pi*0.1*x)
        z = np.linspace(1,2,fps)
        t_rec = np.arange(0.2,1.2,1.0/fps)

        # recreate acoustic tracking assignment :
        t_emit = np.arange(0.1,1.1,0.1)
        x_act = np.copy(x) + np.random.normal(0,0.05,x.size)
        y_act = np.copy(y) + np.random.normal(0,0.05,x.size)
        z_act = np.copy(z) + np.random.normal(0,0.1,z.size)







class TestFindClosestTrajectory(unittest.TestCase):
    '''
    '''

    def setUp(self):
            #create a video based trajectory :
            fps = 25
            self.labld_data = {
            'x': np.linspace(0,1,fps),
            'y': np.sin(2*np.pi*0.1*np.linspace(0,1,fps)),
            'z' : np.linspace(1,2,fps),
            't' : np.arange(0.2,1.2,1.0/fps),
            'traj_num' : np.concatenate( (np.tile(0,12), np.tile(1,13)))
            }

            self.labld_pts = pd.DataFrame(data=self.labld_data)

    def test_singlept_w_singletraj(self):

        focal_ptdata = {
        'x':[self.labld_data['x'][2] ],
        'y': [self.labld_data['y'][2]],
        'z':[self.labld_data['z'][2] +0.01],
        't' :  [self.labld_data['t'][2] + 0.0] }


        focal_df = pd.DataFrame(data=focal_ptdata)

        closestxyz , traj_num = find_closest_trajectory(focal_df,
                                                        self.labld_pts, 0.02)







if __name__ == '__main__':
    unittest.main()

