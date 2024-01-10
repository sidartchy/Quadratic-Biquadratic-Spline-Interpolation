def biquadratic_spline_interpolation(image, scale_factor=2):
  image = np.array(image) # Making sure the image is a numpy array , typecasting if not


  # Performing quadratic interpolation on each row, scaling the columns
  image_new_row = []
  for i in range(image.shape[0]):  ## iterate row wise
    x_data = np.arange(image.shape[1])
    y_data = image[i]
    x_new_row = np.linspace(0,image.shape[1]-1, scale_factor*image.shape[1] )
    quadratic_interpolation= interp1d(x_data, y_data)
    x_pred= quadratic_interpolation(x_new_row)
    image_new_row.append(x_pred)

  image_new_row = np.array(image_new_row)


  # Performing quadratic interpolation on each column, scaling the rows
  final_image = []
  for i in range(image_new_row.shape[1]):  # iterating columnwise
    x_data = np.arange(image_new_row.shape[0])
    y_data = image_new_row.T[i]
    x_new_column = np.linspace(0,image_new_row.shape[0]-1,scale_factor*image_new_row.shape[0])
    quadratic = interp1d(x_data, y_data)
    x_pred = quadratic(x_new_column)
    final_image.append(x_pred)

  final_image = np.array(final_image).T

  return final_image
