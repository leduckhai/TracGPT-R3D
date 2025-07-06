# import torch

# def normalize_boxes(self, boxes,x_min,x_max,y_min,y_max,z_min,z_max):
#         """
#         Normalize boxes to [0, 1] range
#         Args:
#             boxes: [..., 6] - [center_x, center_y, center_z, width, height, length]
#         Returns:
#             normalized_boxes: [..., 6] - normalized to [0, 1]
#         """
#         normalized = boxes.clone()

#         # Normalize centers to [0, 1]
#         # x_range = self.coord_bounds["x_max"] - self.coord_bounds["x_min"]
#         # y_range = self.coord_bounds["y_max"] - self.coord_bounds["y_min"]
#         # z_range = self.coord_bounds["z_max"] - self.coord_bounds["z_min"]
        
#         x_range = x_max - x_min 
#         y_range = y_max - y_min
#         z_range = z_max - z_min
#         normalized[..., 0] = (
#             boxes[..., 0] - self.coord_bounds["x_min"]
#         ) / x_range  # center_x
#         normalized[..., 1] = (
#             boxes[..., 1] - self.coord_bounds["y_min"]
#         ) / y_range  # center_y
#         normalized[..., 2] = (
#             boxes[..., 2] - self.coord_bounds["z_min"]
#         ) / z_range  # center_z

#         # Normalize dimensions by the respective ranges
#         normalized[..., 3] = boxes[..., 3] / x_range  # width
#         normalized[..., 4] = boxes[..., 4] / y_range  # height
#         normalized[..., 5] = boxes[..., 5] / z_range  # length

#         return normalized



# def convert_model_to_gt_format(self, model_boxes,normalize=True):
#         """
#         Convert model predictions from [center_x, center_y, center_z, width, height, length]
#         to ground truth format [x_min, y_min, z_min, x_max, y_max, z_max]

#         Args:
#             model_boxes: [..., 6] - boxes in center+size format
#         Returns:
#             gt_boxes: [..., 6] - boxes in min/max format
#         """
#         # Denormalize if needed
#         if normalize:
#             model_boxes = denormalize_boxes(model_boxes)

#         # Extract center and dimensions
#         center_x, center_y, center_z = (
#             model_boxes[..., 0],
#             model_boxes[..., 1],
#             model_boxes[..., 2],
#         )
#         width, height, length = (
#             model_boxes[..., 3],
#             model_boxes[..., 4],
#             model_boxes[..., 5],
#         )

#         # Convert to min/max coordinates
#         x_min = center_x - width / 2
#         y_min = center_y - height / 2
#         z_min = center_z - length / 2

#         x_max = center_x + width / 2
#         y_max = center_y + height / 2
#         z_max = center_z + length / 2

#         # Stack into GT format
#         gt_boxes = torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], dim=-1)

#         return gt_boxes
