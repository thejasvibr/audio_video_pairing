# -*- coding: utf-8 -*-
""" Tests for the AV_pairing module
Created on Sat Feb 24 21:27:33 2018

@author: tbeleyur
"""

import unittest
import matplotlib.pyplot as plt
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
    '''TODO : write the tests!! !!
    '''


class TestAssignCallstoTrajs(unittest.TestCase):
    '''

    '''
    def setUp(self):
        '''Generate one basic trajectory of a diagonal
        straight line in the xy plane
        '''

        self.fps = 50
        self.traj1 = {
        'x': np.linspace(1,0,self.fps),
        'y': np.linspace(0,1,self.fps),
        'z' : np.linspace(1,2,self.fps),
        't' : np.arange(0.0,1.0,1.0/self.fps),
        'traj_num' : np.tile(0,self.fps)
        }

        self.labld_traj1 = pd.DataFrame(data=self.traj1)

    def test_simultaneous_multipleassignment(self):
        '''The unknown point lies between two sides of an isoceles
        triangle. The closest points are symmetrically located from the
        focal point.
        '''

        self.traj2 = {
        'x': np.linspace(-1,0,self.fps),
        'y': np.linspace(0,1,self.fps),
        'z': np.linspace(1,2,self.fps),
        't': np.arange(0,1,self.fps**-1),
        'traj_num': np.tile(1,self.fps)
        }
        self.labld_traj2 = pd.DataFrame(data=self.traj2)

        combined_trajs = pd.concat([self.labld_traj1, self.labld_traj2])

        unlab_pt = {'x':[0.0,0.0],'y':[0.9,0.99],'z':[1.99,2.0],'t':[0.95,0.96]}
        self.unlab_traj = pd.DataFrame(data=unlab_pt)

        string_ps = {'time_win':0.05,'prox_thres':0.2}

        assigned_df = assign_to_trajectory(combined_trajs, self.unlab_traj,
                                                                     string_ps)
        # check if an equal number of traj 0 and 1 have been calculated.
        output_trajs, num_entries = np.unique(assigned_df['traj_num'], return_counts =True)

        self.assertEqual(num_entries[0], num_entries[1])



class TestFindClosestTrajectory(unittest.TestCase):
    '''
    All the tests to be built in :

    1) Assign a single unlabelled point to a single trajectory in the vicinty

    2.1) Assign a single unlabelled point to multiple trajectories that match
       equally well

    2.2) Assign a single unlabelled point to NO trajectories - because none
       are within the expected range of the point.

    '''

    def setUp(self):
            #create a video based trajectory :
            self.fps = 25
            self.labld_data = {
            'x': np.linspace(0,1,self.fps),
            'y': np.sin(2*np.pi*np.linspace(0,1,self.fps)),
            'z' : np.linspace(1,2,self.fps),
            't' : np.arange(0.0,1.0,1.0/self.fps),
            'traj_num' : np.tile(0,25)
            }

            self.labld_pts = pd.DataFrame(data=self.labld_data)

    def test_singlept_w_singletraj(self):
        ''' Create a single unlabelled point that is very close to one of
        the actual points in the labelled trajectory.
        '''

        labeld_trajptnum = 2
        focal_ptdata = {
        'x':[self.labld_data['x'][labeld_trajptnum] ],
        'y': [self.labld_data['y'][labeld_trajptnum]],
        'z':[self.labld_data['z'][labeld_trajptnum] +0.01],
        't' :  [self.labld_data['t'][labeld_trajptnum] + 0.0], }


        focal_df = pd.DataFrame(data=focal_ptdata)

        closestxyzt , traj_num = find_closest_trajectory(focal_df,
                                                        self.labld_pts, 0.02)


        #check if the closest xyz matches the actual one that was used.
        orig_point = self.labld_pts[['x','y','z','t']].iloc[labeld_trajptnum,:]
        closest_pointmatches = np.array_equal( np.array(orig_point),
                                             np.array(closestxyzt).flatten())
        self.assertTrue(closest_pointmatches)

        self.assertEqual(traj_num.iloc[0], 0 )

    def test_singplt_w_multitrajs(self):
        '''Check if the radial threshold works by assigning a circular path
        to 4 different trajectories.

        With a generous proximity threshold all trajectories should show up.

        With a very small proximity threshold - noe of the trajectories should
            show up.

        '''
        theta = np.linspace(0,2*np.pi,self.fps)
        radius = 2.0
        labld_data = {
        'x' : radius*np.cos(theta),
        'y' : radius*np.sin(theta),
        'z' : np.tile(1,self.fps),
        't' : np.linspace(0,1,self.fps),
        'traj_num' : np.concatenate((np.tile(0,6),np.tile(1,6),np.tile(2,6),
                                                               np.tile(3,7)))
        }
        labld_trajs = pd.DataFrame(data=labld_data)

        focal_data = {'x':[0],'y':[0],'z':[1],'t':[0.2]}
        focal_pt = pd.DataFrame(data=focal_data)


        # set a generous proximity threshold - all points should be covered here
        all_pts, all_inds = find_closest_trajectory(focal_pt, labld_trajs,
                                                            radius+0.5)

        self.assertEqual(all_pts.shape[0],theta.size)

        alltrajs_present = np.array_equal(np.unique(all_inds),
                                                          np.array([0,1,2,3]))
        self.assertTrue(alltrajs_present)

        # set a very narrow proximity threshold - no points should be there :
        closest_pts, closest_inds = find_closest_trajectory(focal_pt, labld_trajs,
                                                            radius-0.5)
        self.assertEqual(closest_pts.shape[0], 0)

    def test_NaNs_in_labldpts(self):
        '''it often happens that there are missing values in the labelled
        trajectory dataset -test if find_closest_trajectory is robust to these
        entries
        '''

        self.labld_pts.iloc[:3][['x','y','z']] = np.nan

        point_number = 5
        focal_pt = self.labld_pts.iloc[point_number][['x','y','z','t']]

        closest_pts, closest_traj = find_closest_trajectory(focal_pt,
                                                           self.labld_pts,0.01)

        points_match = np.array_equal(np.array(closest_pts).flatten(),
                                                          np.array(focal_pt))
        self.assertTrue(points_match)


    def test_shape_of_output_pts(self):
        ''' find_closest_trajectory should give a 1x4 shape pd.DataFrame
        '''

        t = np.linspace(0,1,self.fps)
        self.traj2 = {
        'x': t,
        'y': np.sin(2*np.pi*t),
        'z': np.linspace(1,2,self.fps),
        't': np.arange(0,1,self.fps**-1),
        'traj_num': np.tile(1,self.fps)
        }
        self.labld_traj2 = pd.DataFrame(data=self.traj2)


        unlab_pt = {'x':[0.9],'y':[0.01],'z':[1.8],'t':[0.9]}
        self.unlab_traj = pd.DataFrame(data=unlab_pt)

        string_ps = {'time_win':0.1,'prox_thres':0.3}

        closest_pts, closest_trajs = find_closest_trajectory(self.unlab_traj,
                                    self.labld_traj2, string_ps['prox_thres'])






class Testconvert_DLTdv5_to_xyz(unittest.TestCase):

    def test_basictest(self):

        dltdv5_data = {
        'pt1_X':[0,1,2],
        'pt1_Y':[0,1,2],
        'pt1_Z':[0,1,2],
        'pt2_X':[1,1,2],
        'pt2_Y':[1,1,2],
        'pt2_Z':[1,1,2],
        'pt3_X':[2,1,2],
        'pt3_Y':[2,1,2],
        'pt3_Z':[2,1,2],
        't_rec':[0.1,1.5,2.0]
        }
        mock_dltdv5 = pd.DataFrame(data=dltdv5_data)

        post_conv = convert_DLTdv5_to_xyz(mock_dltdv5)

        numrows, numcols = post_conv.shape

        numpoints = 3
        self.assertEqual(numrows, len(dltdv5_data['t_rec'])*numpoints)
        self.assertEqual(numcols, 5)



class TestCreateCandidatepointDf(unittest.TestCase):
    '''
    '''

    def test_formultiple_candidates(self):
        unlab_pt = pd.DataFrame(data={'x':[0.2],'y':[0.5],'z':[1.5],'t':[0.4]})

        cand_pts = pd.DataFrame(data={'x':[0,1],'y':[2,3],'z':[4,5],
                                                              't':[0.45,0.48]})
        cand_trajs = pd.DataFrame(data={'traj_num':[0,1]})

        cand_df = create_candidatepoint_df(unlab_pt, cand_pts, cand_trajs)
        numrows, numcols = cand_df.shape

        self.assertEqual(numrows, cand_pts.shape[0])
        self.assertEqual(numcols, cand_pts.shape[1] + unlab_pt.shape[1]+1)

    def test_forsinglecandidate(self):

        unlab_pt = pd.DataFrame(data={'x':[0.2],'y':[0.5],'z':[1.5],'t':[0.4]})

        cand_pts = pd.DataFrame(data={'x':[0],'y':[2],'z':[4],
                                                              't':[0.45]})
        cand_trajs = pd.DataFrame(data={'traj_num':[0]})

        cand_df = create_candidatepoint_df(unlab_pt, cand_pts, cand_trajs)
        numrows, numcols = cand_df.shape

        self.assertEqual(numrows, cand_pts.shape[0])
        self.assertEqual(numcols, cand_pts.shape[1] + unlab_pt.shape[1]+1)


    def test_withNaNclosestpoints(self):
        ''' test that no errors are thrown if there are no trajectories
        close by.
        '''

        unlab_pt = pd.DataFrame(columns=['x','y','z','t'])
        cand_pts = pd.DataFrame(columns=['x','y','z','t'])
        cand_trajs = pd.DataFrame(columns=['traj_num'])

        cand_df = create_candidatepoint_df(unlab_pt, cand_pts, cand_trajs)



class TestCalcRadialCI(unittest.TestCase):
    '''
    '''

    def setUp(self):
        self.threerows_data = {'x':[0,0,1],'y':[1,0,0],'z':[0,1,0]}

    def test_validCIs(self):


        xyzCI = pd.DataFrame(data=self.threerows_data)
        xyzCI['radial_CI'] = xyzCI.apply(calc_radial_CI,1)

        radialCI_same = np.array_equal(np.array(xyzCI['radial_CI']).flatten(),
                        np.array([1,1,1]))

        self.assertTrue(radialCI_same)

    def test_wholerowNaN(self):
        for each_axis in ['x','y','z']:
            self.threerows_data[each_axis].append(np.nan)

        xyzCI = pd.DataFrame(data=self.threerows_data)
        xyzCI['radial_CI'] = xyzCI.apply(calc_radial_CI,1)

        na_row = np.array(xyzCI.iloc[-1,:] ).flatten()

        self.assertTrue(sum(np.isnan(na_row)),4)

    def test_oneaxisNaN(self):
        for each_axis in ['x','y','z']:
            self.threerows_data[each_axis].append(0.5)

        self.threerows_data['x'][-1] = np.nan

        xyzCI = pd.DataFrame(data=self.threerows_data)
        xyzCI['radial_CI'] = xyzCI.apply(calc_radial_CI,1)

        self.assertTrue(np.isnan(xyzCI.iloc[-1,-1]))









if __name__ == '__main__':
    unittest.main()

