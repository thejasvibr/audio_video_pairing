# -*- coding: utf-8 -*-
""" Functions which helps in dealing with synchronised audio-video recordings

Created on Sat Feb 24 17:43:26 2018

@author: tbeleyur
"""

import numpy as np
import pandas as pd
import re
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




def convert_DLTdv5_to_xyz(DLTdv5_pt):
    '''Converts the slightly modified outputs of the DLTdv5 trajectory output
    into a long form data set with x y z t columns.

    Parameters:

        DLTdv5_pt : pd.DataFrame with following columns:
            pt1_X,pt1_Y,pt1_Z....ptN_X,ptN_Y,ptN_Z, t_rec

            pt1_X, pt1_Y, pt1_Z : contain the xyz coordinates for each trajectory
                        In the DLTdv5 system, each point number refers to a
                        single trajectory.

            t_rec : float. This is the time-stamp of frame capture with
                    reference to the synchronised audio recording.

    Returns:

        DLTDv5xyz_t : pd.DataFrame. A long version of the raw pd.DataFrame
                    with the following columns:

                x ,y, z : the xyz coordinates of each trajectories points

                traj_num : int. the trajectory number

                t : float. The time of recording of the video frame.

    '''
    DLTdv5_xyz = DLTdv5_pt.drop('t_rec',axis=1)
    numrows, numcols = DLTdv5_xyz.shape

    if numcols % 3 != 0 :
        raise IndexError('Number of columns not a multiple of 3 ! : ' + str(numcols))

    container_pd = pd.DataFrame()
    for each_point in range(1, int(numcols/3.0)+1):

        colnames = [ 'pt'+str(each_point)+'_'+each_axis for each_axis in ['X','Y','Z']   ]
        colnames.append('t_rec')

        pt_trajs = DLTdv5_pt[colnames]
        pt_trajs['traj_num'] = each_point
        pt_trajs.columns = ['x','y','z','t','traj_num']

        container_pd = pd.concat([container_pd,pt_trajs])

    container_pd = container_pd.reset_index(drop=True)

    return(container_pd)



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


def assign_to_trajectory(knwn_trajpts, unknwn_trajpts, stringency_params):
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

        stringency_params: dictionary. Parameters that decide how close the
                            unlabelled points need to be in space and time to
                            be assigned to a given trajectory.

                time_win : float. The time window in seconds to consider
                              while trying to match the unlabelled points to
                              a trajectory.

                            Setting this too wide means that there will be pot-
                            entially many matches. Setting this too narrow coul
                            -d mean that in case there are no points in that t-
                            ime-window, then no trajectory is labelled !

                prox_thres : float. The proxity threshold is the maximum eucli
                            dean distance permitted between the unlabelled
                            points and the labelled trajectories.



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

    num_unknpts, _ = unknwn_trajpts.shape

    for row_num, row in enumerate(unknwn_trajpts.itertuples(),1):

        unlabd_data = np.array([row.x,row.y,row.z,row.t]).reshape((1,4))
        unlabd_point = pd.DataFrame(data=unlabd_data)
        unlabd_point.columns=['x','y','z','t']


        knwn_inwindow = choose_knownpts_intimewindow(knwn_trajpts,
                                         unlabd_point['t'],
                                                stringency_params['time_win'])

        closest_knwnpts, closest_trajs = find_closest_trajectory(unlabd_point,
                                                knwn_inwindow,
                                            stringency_params['prox_thres'])


        labeld_and_candidate_df = create_candidatepoint_df(unlabd_point,
                                               closest_knwnpts,  closest_trajs)

        labeled_pts = labeled_pts.append(labeld_and_candidate_df,
                                                             ignore_index=True)

    return(labeled_pts)



def choose_knownpts_intimewindow(knwn_pts, time_stamp, time_window):
    '''Chooses a subset of a pd.DataFrame with time that falls within
    a given time stamp.

    Eg. If the time_stamp is 0.2 , and the time_window is 0.1, then all
    rows with      0.15<=knwn_pts['t']<=0.25 will be returned.

    Parameters:

        knwn_pts : pd.DataFrame with at least the following columns:
            x: x-coordinate
            y: y-coordinate
            z: z-coordinate
            t: time

        time_stamp : float. Time point around which the time window is centred.

        time_window : float >0. The length of the time window around the time
                        stamp.


    Returns:

        knwn_inwin : pd.DataFrame. A subset of knwn_pts rows that fall within
                     the time window centred on the time stamp. This can also
                     be an empty row if there are no rows that fulfil this
                     condition.

    '''
    if time_window <= 0 :
        raise ValueError('Time window length cannot be <= 0! ')

    win_start = float(time_stamp) - time_window/2.0
    win_end = float(time_stamp) + time_window/2.0

    knwn_inwin = knwn_pts[(knwn_pts['t']>= win_start) & (knwn_pts['t']<= win_end) ]

    return(knwn_inwin)

def find_closest_trajectory(focal_pt, labld_pts, dist_threshold):
    ''' Finds the closest labelled point to a focal point and assigns it
    the trajectory number.

    Parameters:

        focal_pt : 1 x 4 pd.DataFrame with the following column names:
                    x, y, z, t

        labld_pts : Npoints x 5 pd.DataFrame with the following column names:
                    x , y, z, t, traj_num


    Returns :

        closest_labdpt : pd.DataFrame. with the closest labelled point/s

        traj_num: pd.DataFrame. label number/s of the known trajectory

    '''
    distances = labld_pts[['x','y','z']].apply(calc_euc_distance,1,
                                                v= focal_pt[['x','y','z']])
    closest_ptind = distances <= dist_threshold

    traj_num = labld_pts['traj_num'][closest_ptind].reset_index(drop=True)

    closest_labdpt = labld_pts[['x','y','z','t']][closest_ptind].reset_index(drop=True)

    return(closest_labdpt, traj_num)


def calc_euc_distance(u,v):
    ''' A wrapper function which calculates euclidean distance
    between two points - *and* accounts for the possibility that
    in case one set of points has NaNs - then it returns a Nan!

    This was written so that the distance.euclidean function from
    scipy.spatial could be applied across rows of a pd.DataFrame
    '''
    try:
        d = spatial.distance.euclidean(u,v)

    except:
        d = np.nan

    return(d)

def create_candidatepoint_df(unlab_pt, candidate_points, trajs):
    '''Formatting function which replicates the unlabeled point over multiple
    rows and joins it with the candidate points

    Parameters:

        unlab_pt : 1x 4 pd.DataFrame with the following columns:
                    x, y, z ,t

        candidate_points : Npoints x 4 pd.DataFrame with folowing columns :
                    x, y, z, t

        trajs : Npoints x 1 pd.DataFrame with trajectory numbers of each of the
                candidate points.


    Returns:

        candidatepts_df : Npoints x 9 pd.DataFrame with the following columns:
                    x,y,z : xyz coordinates of the unlabelled point
                    t : time of recording/emission of the unlabelled point
                    traj_num : candidate trajectory number of the unlabelled
                               points
                   x_knwn, y_knwn, z_knwn : xyz coordinates of the known
                                trajectory points
                    t_knwn : time of emission/recording of the known trajectory
                             point
    '''

    num_candpoints, _ = candidate_points.shape



    if num_candpoints >1:

        points_and_trajs = candidate_points.join(trajs)
        repeated_unlabpts = unlab_pt.append([unlab_pt]*(num_candpoints-1),
                                            ignore_index=True)

        try:
            candidatepts_df = repeated_unlabpts.join(points_and_trajs,
                                                               rsuffix='_knwn')
        except:
            repeated_unlabpts = repeated_unlabpts.to_frame()
            candidatepts_df = repeated_unlabpts.join(points_and_trajs,
                                                               rsuffix='_knwn')

    else:
        candidatepts_df = unlab_pt.join(trajs).join(candidate_points,
                                                               rsuffix='_knwn')
    return(candidatepts_df)

