# -*- coding: utf-8 -*-
""" Functions which helps in dealing with synchronised audio-video recordings

Created on Sat Feb 24 17:43:26 2018

@author: tbeleyur
"""

import numpy as np
import pandas as pd
import scipy.spatial as spatial

def choose_reliable_points(vidpoints, actpoints, threshold=0.1):
    '''Gives reliable points that have similar euclidean distances
    to each other in the video tracking and acoustic tracking frames of
    reference.

    This function works on the assumption that pairs of points that have simi-
    lar distances to each other in video tracking and acoustic tracking
    are reliable points.

    Even though the euclidean distances of all point-pairs may not be the
    same - this function at least removes those points that are completely
    off.

    The input xyz points should be somehow time-synchronised to avoid large
    errors in estimation.

    TODO :
    1) add in checks for dimension of vidpoints and actpoints
    2) check if there are common points in vid and act points
     -  and throw a warning

    Parameters:

        vidpoints : Npoints x 3 np.array or pd.DataFrame with xyz coordinates

        actpoints : Npoints x 3 np.array or pd.DataFrame with xyz coordinates

        threshold : float. Maximum deviation between point distances.
                    Any pair distances that have lower or equal than threshold
                    estimate error between the two systems will be kept.

    Returns:

        reliable_inds : np.array. Indices of the points that have similar
                        euclidean distances to each other in both tracking
                        systems.
    Note :
    This function may output misleading results if there are the exact same
    points in vidpoints and actpoints


    '''
    # add in checks to see if threshold is +ve and that the shape of the xyz
    # points are same

    # calculate distance matrices for both system's points :
    vid_distmat = spatial.distance_matrix(vidpoints, vidpoints)
    act_distmat = spatial.distance_matrix(actpoints, actpoints)

    # deviation between pairs of points in both tracking systems
    diff_distmats = abs( vid_distmat - act_distmat )

    # choose best tracking points :
    pair_dists = diff_distmats[np.tril_indices_from(diff_distmats,-1)]
    good_inds = np.where(pair_dists <= threshold)
    good_dists = pair_dists[good_inds]

    point_indices = []
    tril_distmat = np.tril(diff_distmats,-1)

    for each_gooddist in good_dists:
        row, col = np.where(each_gooddist == tril_distmat)
        point_indices.append(row)
        point_indices.append(col)

    indices = np.unique(np.concatenate(point_indices))

    return(indices, good_dists)


def perform_rigidtransform( xyzpts, trform_mat):
    ''' Performs a rigid transform on points in the a reference frame
    into another using a given transform matrix.



    Parameters :

       xyzpts : Npoints x 3 np.array. The raw xyz coordinates of the points.

       trform_mat : 4 x 4 np.array. The transform matrix

     Returns :

         trformd_pts : Npoints x 3 np.array. xyzpts post rigid transformation
    '''

    xyz_homog = np.ones((xyzpts.shape[0],4))
    xyz_homog[:,:3] = xyzpts
    trformd_mat = np.dot(trform_mat, xyz_homog.T).T
    trformd_pts = trformd_mat[:,:3]

    return(trformd_pts)


def assign_to_trajectory(knwn_trajpts, unknwn_trajpts, time_window=0.1):
    '''Assigns a set of unknown points to a set of known labelled trajectories.

TODO :
1) add assertions for the shape and size of the inputs !
2) what happens if there are > 1 closest trajectories ?

    Parameters:

        knwn_trajpts : pd.DataFrame with following columns:
            x, y, z, t, traj_number - where:

                x , y, z : these are the xyz coordiantes in the trajectory

                t : float. time of recording/emission

                traj_number : int. The trajectory label.

        unknwn_trajpts : pd.DataFrame with the unlabeled points. It
                        has the following columns:
            x, y, z, t - see above for the description of columns.


        time_window : float. The time window to consider while trying to -
                        match the unlabelled points to a trajectory.

                        Setting this too wide means that there will be potenti-
                        ally many matches. Setting this too narrow could
                        mean that in case there are no points in that time-win-
                        dow, then no trajectory is labelled !

    Returns :

        labelled_trajpts : pd.DataFrame with following columns:
            x , y, z, t, closest_traj, x_knwn, y_knwn, z_knwn, t_knwn
            where :

                x, y, z : coordinates of the unlabelled points

                t : time of recording/emission

                closest_traj : int. closest labelled trajectory in time and
                                space.
                x_knwn, y_knwn, z_knwn : coordinates of the trajectory-labelled
                        point closest to the unlabelled point.
                        to the
                t_knwn : float. time of recording/emission in the labelled
                         trajectory.d
    '''

    labeled_pts = pd.DataFrame()

    for row_num, unlabd_point in unknwn_trajpts.iterrows():

        knwn_inwindow = choose_knownpts_intimewindow()

        closest_knwnpt, closest_trajs = find_closest_trajectory(knwn_inwindow,
                                                                unlabd_point)

        labeled_pts.append([unlabd_point, closest_trajs, closest_knwnpt])

    #return(labeled_pts)
    pass


def choose_knownpts_intimewindow(knwn_pts, time_window):
    '''
    '''
    win_start = knwn_pts['t']-time_window/2.0
    win_end = knwn_pts['t']+time_window/2.0

    knwn_inwin = knwn_pts[ (knwn_pts['t']>= win_start) &((knwn_pts['t']<= win_end)) ]

    return(knwn_inwin)

def find_closest_trajectory(focal_pt, labld_pts, dist_threshold):
    ''' Finds the closest labelled point to a focal point and assigns it
    the trajectory number.

    Parameters:

        focal_pt : 1 x 4 pd.DataFrame with the following column names:
                    x, y, z, t

        labld_pts : Npoints x 5 pd.DataFrame with the following column names:
                    x , y, z, t, traj_number


    Returns :

        closest_labdpt : pd.DataFrame. with the closest labelled point

        traj_num: pd.DataFrame. label number of the known trajectory

    TODO :
        1) Take care of multiple-matching case
        2) What happens if a trajectory number is a NaN

    '''
    distances = labld_pts[['x','y','z']].apply(spatial.distance.euclidean,1,
                                                v= focal_pt[['x','y','z']])
    closest_ptind = distances <= dist_threshold

    traj_num = labld_pts['traj_num'][closest_ptind]
#    if np.isnan(traj_num):
#        raise ValueError('Unassigned trajectory found')

    closest_labdpt = labld_pts[['x','y','z']][closest_ptind]

    return(closest_labdpt, traj_num)


