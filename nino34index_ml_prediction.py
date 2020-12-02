# %matplotlib inline
import xarray as xr
import pandas as pd
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import sys
import matplotlib.pyplot as plt
import sklearn

#Scaffold code to load in data.  This code cell is mostly data wrangling


def load_enso_indices():
  """
  Reads in the txt data file to output a pandas Series of ENSO vals

  outputs
  -------

    pd.Series : monthly ENSO values starting from 1870-01-01
  """
  # with open('mount/data/nino34.long.anom.data.txt') as f:
  #   line = f.readline()
  #   enso_vals = []
  #   while line:
  #       yearly_enso_vals = map(float, line.split()[1:])
  #       enso_vals.extend(yearly_enso_vals)
  #       line = f.readline()

  # enso_vals = pd.Series(enso_vals)
  # enso_vals.index = pd.date_range('1870-01-01',freq='MS',
  #                                 periods=len(enso_vals))
  # enso_vals.index = pd.to_datetime(enso_vals.index)
  # return enso_vals

def assemble_predictors_predictands(start_date, end_date, lead_time,
                                    input_data, data_format,
                                    num_input_time_steps=1,
                                    use_pca=False, n_components=32,
                                    lat_slice=None, lon_slice=None):
  """
  inputs
  ------

      start_date           str : the start date from which to extract sst
      end_date             str : the end date
      lead_time            str : the number of months between each sst
                              value and the target Nino3.4 Index
      dataset              str : 'observations' 'CNRM' or 'MPI'
      data_format          str : 'spatial' or 'flatten'. 'spatial' preserves
                                  the lat/lon dimensions and returns an
                                  array of shape (num_samples, num_input_time_steps,
                                  lat, lon).  'flatten' returns an array of shape
                                  (num_samples, num_input_time_steps*lat*lon)
      num_input_time_steps int : the number of time steps to use for each
                                 predictor sample
      use_pca             bool : whether or not to apply principal components
                              analysis to the sst field
      n_components         int : the number of components to use for PCA
      lat_slice           slice: the slice of latitudes to use
      lon_slice           slice: the slice of longitudes to use

  outputs
  -------
      Returns a tuple of the predictors (np array of sst temperature anomalies)
      and the predictands (np array the ENSO index at the specified lead time).

  """
  file_name = 'mount/data/{}'.format(input_data)
              #  'observations2': 'mount/data/regridded_era_t2m_anomalies.nc',
              #  'CNRM'         : 'mount/data/CNRM_tas_anomalies_regridded.nc',
              #  'MPI'          : 'mount/data/MPI_tas_anomalies_regridded.nc'}[dataset]
  # variable_name = {'observations' : 'sst',
  #                  'observations2': 't2m',
  #                  'CNRM'         : 'tas',
  #                  'MPI'          : 'tas'}[dataset]
  ds = xr.open_dataset(file_name)
  variable_name = list(ds.data_vars)[0]
  sst = ds[variable_name].sel(time=slice(start_date, end_date))
  if lat_slice is not None:
    try:
        sst=sst.sel(lat=lat_slice)
    except:
        raise NotImplementedError("Implement slicing!")
  if lon_slice is not None:
    try:
        sst=sst.sel(lon=lon_slice)
    except:
        raise NotImplementedError("Implement slicing!")


  num_samples = sst.shape[0]
  #sst is a (num_samples, lat, lon) array
  #the line below converts it to (num_samples, num_input_time_steps, lat, lon)
  sst = np.stack([sst.values[n-num_input_time_steps:n] for n in range(num_input_time_steps,
                                                              num_samples+1)])
  #CHALLENGE: CAN YOU IMPLEMENT THE ABOVE LINE WITHOUT A FOR LOOP?
  num_samples = sst.shape[0]

  sst[np.isnan(sst)] = 0
  if data_format=='flatten':
    #sst is a 3D array: (time_steps, lat, lon)
    #in this tutorial, we will not be using ML models that take
    #advantage of the spatial nature of global temperature
    #therefore, we reshape sst into a 2D array: (time_steps, lat*lon)
    #(At each time step, there are lat*lon predictors)
    sst = sst.reshape(num_samples, -1)


    #Use Principal Components Analysis, also called
    #Empirical Orthogonal Functions, to reduce the
    #dimensionality of the array
    if use_pca:
      pca = sklearn.decomposition.PCA(n_components=n_components)
      pca.fit(sst)
      X = pca.transform(sst)
    else:
      X = sst
  else: # data_format=='spatial'
    X = sst

  start_date_plus_lead = pd.to_datetime(start_date) + \
                        pd.DateOffset(months=lead_time+num_input_time_steps-1)
  end_date_plus_lead = pd.to_datetime(end_date) + \
                      pd.DateOffset(months=lead_time)
  # if dataset == 'observations':
  #   y = load_enso_indices()[slice(start_date_plus_lead,
  #                                 end_date_plus_lead)]
  # else: #the data is from a GCM
  if True:
    X = X.astype(np.float32)
    #The Nino3.4 Index is composed of three month rolling values
    #Therefore, when calculating the Nino3.4 Index in a GCM
    #we have to extract the two months prior to the first target start date
    target_start_date_with_2_month = start_date_plus_lead - pd.DateOffset(months=2)
    subsetted_ds = ds[variable_name].sel(time=slice(target_start_date_with_2_month,
                                                   end_date_plus_lead))
    #Calculate the Nino3.4 index
    y = subsetted_ds.sel(lat=slice(5,-5), lon=slice(360-170,360-120)).mean(dim=('lat','lon'))

    y = pd.Series(y.values).rolling(window=3).mean()[2:].values
    y = y.astype(np.float32)
  ds.close()

  return X.astype(np.float32), y.astype(np.float32)


class ENSODataset(Dataset):
    def __init__(self, predictors, predictands):
        self.predictors = predictors
        self.predictands = predictands
        assert self.predictors.shape[0] == self.predictands.shape[0], \
               "The number of predictors must equal the number of predictands!"

    def __len__(self):
        return self.predictors.shape[0]

    def __getitem__(self, idx):
        return self.predictors[idx], self.predictands[idx]

def plot_nino_time_series(y, predictions, title):
  """
  inputs
  ------
    y           pd.Series : time series of the true Nino index
    predictions np.array  : time series of the predicted Nino index (same
                            length and time as y)
    titile                : the title of the plot

  outputs
  -------
    None.  Displays the plot
  """
  predictions = pd.Series(predictions, index=y.index)
  predictions = predictions.sort_index()
  y = y.sort_index()

  plt.plot(y, label='Ground Truth')
  plt.plot(predictions, '--', label='ML Predictions')
  plt.legend(loc='best')
  plt.title(title)
  plt.ylabel('Nino3.4 Index')
  plt.xlabel('Date')
  plt.show()
  plt.close()

class CNN(nn.Module):
    def __init__(self, num_input_time_steps=1, print_feature_dimension=False):
        """
        inputs
        -------
            num_input_time_steps        (int) : the number of input time
                                                steps in the predictor
            print_feature_dimension    (bool) : whether or not to print
                                                out the dimension of the features
                                                extracted from the conv layers
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_input_time_steps, 64, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 64, 5, padding=2)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout2d = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(64 * 990, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.print_feature_dimension = print_feature_dimension
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2d(self.pool(F.relu(self.conv3(x))))
        x = x.view(-1, 64 * 990)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
'''
class CNN(nn.Module):
    def __init__(self, num_input_time_steps=1, print_feature_dimension=False):
        """
        inputs
        -------
            num_input_time_steps        (int) : the number of input time
                                                steps in the predictor
            print_feature_dimension    (bool) : whether or not to print
                                                out the dimension of the features
                                                extracted from the conv layers
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_input_time_steps, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.print_layer = Print()

        #ATTENTION EXERCISE 9: print out the dimension of the extracted features from
        #the conv layers for setting the dimension of the linear layer!
        #Using the print_layer, we find that the dimensions are
        #(batch_size, 16, 42, 87)
        self.fc1 = nn.Linear(16 * 42 * 87, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.print_feature_dimension = print_feature_dimension

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.print_feature_dimension:
          x = self.print_layer(x)
        x = x.view(-1, 16 * 42 * 87)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
'''

class Print(nn.Module):
    """
    This class prints out the size of the features
    """
    def forward(self, x):
        print(x.size())
        return x


def train_network(net, testloader, experiment_name):
  """
  inputs
  ------

      net               (nn.Module)   : the neural network architecture
      criterion         (nn)          : the loss function (i.e. root mean squared error)
      optimizer         (torch.optim) : the optimizer to use update the neural network
                                        architecture to minimize the loss function
      trainloader       (torch.utils.data.DataLoader): dataloader that loads the
                                        predictors and predictands
                                        for the train dataset
      testloader        (torch.utils.data. DataLoader): dataloader that loads the
                                        predictors and predictands
                                        for the test dataset
  outputs
  -------
      predictions (np.array), and saves the trained neural network as a .pt file
  """
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  net = net.to(device)
  best_loss = np.infty


  net = torch.load('mount/model/{}.pt'.format(experiment_name), map_location=torch.device('cpu'))
  net.eval()
  net.to(device)

  #the remainder of this notebook calculates the predictions of the best
  #saved model
  predictions = np.asarray([])
  for i, data in enumerate(testloader):
    batch_predictors, batch_predictands = data
    batch_predictands = batch_predictands.to(device)
    batch_predictors = batch_predictors.to(device)

    batch_predictions = net(batch_predictors).squeeze()
    #Edge case: if there is 1 item in the batch, batch_predictions becomes a float
    #not a Tensor. the if statement below converts it to a Tensor
    #so that it is compatible with np.concatenate
    if len(batch_predictions.size()) == 0:
      batch_predictions = torch.Tensor([batch_predictions])
    predictions = np.concatenate([predictions, batch_predictions.detach().cpu().numpy()])
  # if classification == True:
  #   predictions = np.where(predictions<=0, 0, predictions)
  #   predictions = np.where(predictions>0, 1, predictions)
  return predictions #, train_losses, test_losses

# option user can select
lead_range = int(sys.argv[1])
interest_year = sys.argv[2] # from 2011 to 2018
interest_month = int(sys.argv[3])  #  User can select from '2011-01-01' (2011 January) to '2018-12-31'Actually end date:sst
input_data = sys.argv[4]  # 'observations' or 'observations2'
#### user input ###########

import plotly.graph_objects as go

# user input check
lead_time_check = [i for i in range(1,13)]
year_check = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']
# input_data_check = ['observations', 'observations2']

if lead_range not in lead_time_check:
    print("Please enter an integer number from 1 to 12 for lead time.")
# elif interest_year not in year_check:
#     print("Please enter one year from 2011 to 2018 for your interest year.")  
    
elif interest_month not in lead_time_check:
    print("Please enter a month from 1 to 12 for your interest month.")
    
# elif input_data not in input_data_check:
#     print("Please enter 'observations' or 'observations2' for input data.")

else: 
  

  if interest_month in [1,3,5,7,8,10,12]:
      day = str(31)
  elif interest_month == '02':
      
      if interest_year in ['2012', '2016']:
          day = str(29)
      else:
          day = str(28)
  else:
      day = str(30)
  interest_month = '0'+str(interest_month)    
  test_end_date = interest_year+'-'+interest_month[-2:]+'-'+day 
  
  
  start = '2011-01-01'
  end = interest_year+'-'+interest_month[-2:]+'-'+'01'
  make_month = [0,1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6,7,8,9,10,11,12]

  #Assemble numpy arrays corresponding to predictors and predictands
  train_start_date = '1860-01-01'
  train_end_date = '2200-12-31'
  num_input_time_steps = 2

  predict_start_date = '2011-01-01'
  test_start_date = ['2010-11-01', '2010-10-01', '2010-09-01', '2010-08-01', '2010-07-01', '2010-06-01', \
                    '2010-05-01', '2010-04-01', '2010-03-01', '2010-02-01', '2010-01-01', '2009-12-01']

  lead_time_list = [i for i in range(1, lead_range +1)]

  """update batch_size"""
  batch_size = [16,16,16,16,16,16,16,16,16,16,16,16]

  # climate_model = 'MPI'

  result_dic = dict()
  corr_dic = dict()
  keys = list()

  for lead_time in lead_time_list:

    test_predictors, test_predictands = assemble_predictors_predictands(test_start_date[lead_time-1],
                        test_end_date, lead_time, input_data, 'spatial', num_input_time_steps=num_input_time_steps)

    # test_predictors1, test_predictands1 = assemble_predictors_predictands(test_start_date[lead_time-1],
    #                     test_end_date, lead_time, 'observations2', 'spatial', num_input_time_steps=num_input_time_steps)

    # if input_data == 'observations':
    #   test_predictors, test_predictands = test_predictors0, test_predictands0
    # else:
    #   test_predictors, test_predictands = test_predictors1, test_predictands0

    test_dataset = ENSODataset(test_predictors, test_predictands)


    testloader = DataLoader(test_dataset, batch_size=batch_size[lead_time-1])
    net = CNN(num_input_time_steps=num_input_time_steps)


    experiment_name = "twolayerCNN_CNRM_{}_{}_lead_time{}".format(train_start_date, train_end_date, str(lead_time))
    predictions = train_network(net, testloader, experiment_name)


    corr, _ = pearsonr(test_predictands, predictions)
    rmse = mean_squared_error(test_predictands, predictions) ** 0.5

    # print(test_predictands)

    # making between months
    date_list = list()
    for year in range(int(start.split('-')[0]), int(end.split('-')[0])):
        for month in range(1, 13):
            make_date = str(year)+'-'+str(month)+'-'+'01'
            date_list.append(make_date)
    # making end months
    index0 = make_month.index(int(end.split('-')[1]))
    index1 = index0+lead_time
    if index1 <13:
        for month in range(1, index1+1):
            make_date = str(int(end.split('-')[0]))+'-'+str(month)+'-'+'01'
            date_list.append(make_date)
    else:
        for month in range(1, 13):
            make_date = str(int(end.split('-')[0]))+'-'+str(month)+'-'+'01'
            date_list.append(make_date)
        for month in range(1, index1-11):
            make_date = str(int(end.split('-')[0])+1)+'-'+str(month)+'-'+'01'
            date_list.append(make_date)   

    # predictions = pd.Series(predictions, index=test_predictands.index)
    date_list = pd.to_datetime(date_list)
    # predictions = pd.Series(predictions, index=date_list)
    # predictions = predictions.sort_index()
    #y = test_predictands.sort_index()
    y = pd.Series(test_predictands, index=date_list)
    result = pd.DataFrame(y, columns=["ground_truth"])
    column_name = 'lead time'+str(lead_time)
    result[column_name] = predictions

    # print(result)


    key = "lead time"+str(lead_time)
    keys.append(key)

    corr_dic[key] = [corr, rmse]
    result_dic[key] = result

  title = "{}_prediction_{}_{}_lead time till {}".format(input_data, predict_start_date, test_end_date, str(lead_time))

  df = result_dic[key]
  for i in range(lead_range-1):
    null_list = [None for j in range(lead_range-i-1) ]  
    
    column = 'lead time'+str(i+1)
    df[column] = list(result_dic[keys[i]][column]) +null_list

  # for i in range(2, 13):
  #   if i < 10:
  #     date = '2019-0'+str(i)+'-01'
  #   else:
  #     date = '2019-'+str(i)+'-01'
  #   if date in df.index:
  #     df['ground_truth'][date] = None

  sorted(df.columns[1:])
  # Re-order Columns
  df = df[list(df.columns[:1])+sorted(df.columns[1:])]
  result = df
  # 1. output csv file
  # df.to_csv('{}.csv'.format(title))
  df.to_csv('mount/output/{}.csv'.format(title))

  # 2. output predictions table
  date_list=test_end_date.split('-')
  standard_date = date_list[0]+'-'+date_list[1]+'-'+'01'  # '2018-12-01'
  # the first output display : prediction result
  print(df.loc[standard_date:])

  # the second output: correlation table
  df_corr = pd.DataFrame(corr_dic).T
  df_corr.columns = ["correlation", "rmse"]
  print(df_corr)

  fig = go.Figure()

  fig.add_trace(go.Scatter(x=result.index, y=result.ground_truth,
                      line=dict(color='navy', dash='dash'),
                      name='ground truth'))

  for column in result.columns:
      if column.startswith('lead time'):
            
          fig.add_trace(go.Scatter(x=result.index, y=result[column],
                          mode='lines', #line_color= "magenta",
                          name=column))


  fig.add_shape(type="line",
      x0=result.index[0], y0=1.5, x1=result.index[-1], y1=1.5,
      line=dict(
          color= 'Red', #"LightSeaGreen",
          width=2,
          dash="dashdot",
      )
  )

  fig.add_shape(type="line",
      x0=result.index[0], y0=-1.5, x1=result.index[-1], y1=-1.5,
      line=dict(
          color= 'Red', #"MediumPurple",
          width=2,
          dash="dashdot",
      )
  )

  fig.add_shape(
          type="rect",
          xref="x",
          yref="paper",
          x0=standard_date,
          y0="0",
          x1=result.index[-1],
          y1="1",
          fillcolor="gray",
          opacity=0.4,
          line_width=0,
          layer="below"
      ) 

  fig.update_xaxes(
      rangeslider_visible=True,
      rangeselector=dict(
          buttons=list([
              dict(count=1, label="1y", step="year", stepmode="backward"),
              dict(count=2, label="2y", step="year", stepmode="backward"),
              dict(count=3, label="3y", step="year", stepmode="backward"),
              dict(count=4, label="4y", step="year", stepmode="backward"),
              dict(count=5, label="5y", step="year", stepmode="backward"),
              dict(count=6, label="6y", step="year", stepmode="backward"),
              dict(step="all")
          ]), 
          buttondefaults = dict(count=1, label="1y", step="year", stepmode="backward")
      )
  )

  fig.update_yaxes(title_text="nino 3.4 index")
  fig.update_layout(
    width = 950, title = title , 
  )
  fig.update_layout(hovermode="x unified")


  # layout.template.layout.xaxis.rangeselector.buttondefaults

  fig.show()