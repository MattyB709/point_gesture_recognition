# Store dictionary transformation_map mapping april tag id : rigid transformation (from tag -> world frame)

# Loop through every frame of the video (if doing individual images, we'd just continually loop while images are taken)
    # Have dictionary of tag id : rigid transformation (tag -> camera) for all unknown tags
    # Have camera -> tag transformation for known tag
    # Loop through all the detected tags
        # If trasnformation_map is empty, means we have not processed any tags yet
            # Set the first tag we detect as the world frame (add it to the dictionary, map it to 4x4 identity matrix)
        # Determine the transformation from tag to camera
        # If the tag id is in the dictionary, we know its world frame transformation has already been calculated
            # Save the inverse of the transformation (camera -> tag we know global transform for)
        # Else it's new tag
            # Save the tag id & transformation (tag -> camera) in dict of tags we need to calculate world trasnform for
    # Loop through the dict of tags needing world transform
        # Multiply tag -> camera transform by saved camera -> known tag transform, save as tag -> known tag
        # Multiply tag -> known tag by the known tag -> world frame stored in trasnformation_map (result is the trasnformation of tag -> world frame)
        # Save result in trasnformation_map