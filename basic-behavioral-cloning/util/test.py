import csv
import intera_interface

import rospy

rospy.init_node('test')

limb = intera_interface.Limb('right')


def generate_next_actions():
	source_file = open('0.csv', 'rb')
	reader = csv.reader(source_file)

	dest_file = open('1.csv', 'w')
	writer = csv.writer(dest_file)
	writer.writerow(['right_j0_next','right_j1_next','right_j2_next','right_j3_next','right_j4_next','right_j5_next','right_j6_next, eef_pose_x, eef_pose_y, eef_pose_z'])

	reader.next()
	reader.next()
	reader = list(reader)

	for i in range(len(reader)-1):
		if float(reader[i][9]) == float(reader[i+1][9]):
			joints = {'right_j0':float(reader[i+1][2]), 'right_j1':float(reader[i+1][3]), 'right_j2': float(reader[i+1][4]), 'right_j3': float(reader[i+1][5]), 'right_j4':float(reader[i+1][6]), 'right_j5':float(reader[i+1][7]), 'right_j6':float(reader[i+1][8])}
			eef_pose = limb.joint_angles_to_cartesian_pose(joints)
			writer.writerow([reader[i+1][2], reader[i+1][3], reader[i+1][4], reader[i+1][5], reader[i+1][6], reader[i+1][7], reader[i+1][8], eef_pose.position.x, eef_pose.position.y, eef_pose.position.z])
		else:
			joints = {'right_j0':float(reader[i][2]), 'right_j1':float(reader[i][3]), 'right_j2': float(reader[i][4]), 'right_j3': float(reader[i][5]), 'right_j4': float(reader[i][6]), 'right_j5':float(reader[i][7]), 'right_j6':float(reader[i][8])}
			eef_pose = limb.joint_angles_to_cartesian_pose(joints)
			writer.writerow([reader[i][2], reader[i][3], reader[i][4], reader[i][5], reader[i][6], reader[i][7], reader[i][8], eef_pose.position.x, eef_pose.position.y, eef_pose.position.z])

generate_eef_pose():
	source_file = open('0.csv', 'rb')
	reader = csv.reader(source_file)

	dest_file = open('eef.csv', 'w')
	writer = csv.writer(dest_file)
	writer.writerow(['right_j0','right_j1','right_j2','right_j3','right_j4','right_j5','right_j6, eef_pose_x, eef_pose_y, eef_pose_z'])

	reader.next()
	reader = list(reader)

	for i in range(len(reader)-1):
		joints = {'right_j0':float(reader[i][2]), 'right_j1':float(reader[i][3]), 'right_j2': float(reader[i][4]), 'right_j3': float(reader[i][5]), 'right_j4': float(reader[i][6]), 'right_j5':float(reader[i][7]), 'right_j6':float(reader[i][8])}
		eef_pose = limb.joint_angles_to_cartesian_pose(joints)
		writer.writerow([reader[i][2], reader[i][3], reader[i][4], reader[i][5], reader[i][6], reader[i][7], reader[i][8], eef_pose.position.x, eef_pose.position.y, eef_pose.position.z])

# generate_next_actions()
generate_eef_pose()