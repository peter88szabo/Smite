import os
i = 1
# Check if the initial filename exists
while os.path.exists(traj_file):
    # If the filename contains a number, extract it and increment
    if traj_file.endswith(".xyz"):
        base_name = traj_file[:-(len(".xyz"))]  # Remove the ".xyz" extension
        parts = base_name.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            i = int(parts[1]) + 1
            traj_file = f"{parts[0]}_{i}.xyz"
        else:
            traj_file = f"{parts[0]}_1.xyz"
    else:
        # If there is no extension, add "_1.xyz"
        traj_file = f"{traj_file}_1.xyz"
# Now traj_file contains a non-existing filename
print(traj_file)

