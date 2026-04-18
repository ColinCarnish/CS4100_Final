from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import (
    confusion_matrix
)
from sklearn.utils import resample
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve



class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # weights for all gates combined
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)
    
    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        
        hidden_state = torch.zeros(batch_size, self.hidden_size)
        cur_mem_state = torch.zeros(batch_size, self.hidden_size)
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            combined = torch.cat((hidden_state, x_t), dim=1)
            gates = self.W(combined)
            
            # initializing the activation fucntions used per gate 
            forget, input, convert, output = gates.chunk(4, dim=1)

            forget = torch.sigmoid(forget)
            input = torch.sigmoid(input)
            convert = torch.tanh(convert)
            output = torch.sigmoid(output)
            
            # initializing cur memory cell state and hidden state 
            cur_mem_state = forget * cur_mem_state + input * convert
            hidden_state = output * torch.tanh(cur_mem_state)
            
            outputs.append(hidden_state.unsqueeze(1))
        
        outputs = torch.cat(outputs, dim=1)
        return outputs, (hidden_state, cur_mem_state)
    
class DelayPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = CustomLSTM(input_size, hidden_size) # using LSTM we built from scratch
        self.fc = nn.Linear(hidden_size, 1)  # probability of delay per stop
        self.dropout = nn.Dropout(0.2) # learn relationships instead of memorizing
    
    # predicting delay at each stop
    def forward(self, x):
        outputs, _ = self.lstm(x) # dims (batch, seq len, hidden size)
        outputs = self.dropout(outputs)
        logits = self.fc(outputs)
        return logits.squeeze(-1) # dims (batch, seq len)
    
    
 

day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}  

features = ['stop_sequence', 'day_of_week_num', 'month_num', 'hour', 'route_id_Blue', 'route_id_Red', 'route_id_Orange', 'travel_dur']
target = ['delayed']
scale_cols = ['stop_sequence', 'hour', 'day_of_week_num', 'month_num', 'travel_dur']
# scale the delay values so they are all centered around a specific point and are comparable without some delay values outweighing others
scaler = StandardScaler()
 
def preprocess_for_model():
    data = pd.read_csv('Datasets/final_data.csv')
    # convert day of the week and month to their numerical representations to send to the model
    data["day_of_week_num"] = data["day_of_week"].map(day_map)
    data['month_num'] = data['service_date'].str[5:7].astype(int)
    
    # combined the arrival and departure delay into one row per trip id and stop id
    delay_df = data.pivot_table(
        index=["trip_id", "stop_sequence"],
        columns="event_type",
        values="delay_sec"
    ).reset_index()
    
    time_df = data[[
        "trip_id",
        "stop_sequence",
        "arrival_time_sec",
        "departure_time_sec"
    ]].drop_duplicates()
    
    combined_data = delay_df.merge(
        time_df,
        on=["trip_id", "stop_sequence"],
        how="left"
    )
    
    combined_data = combined_data.sort_values(["trip_id", "stop_sequence"])

    # add a column for computing the arrival and departure delay at the previous stop per entry (delays usually have a domino effect)
    combined_data["PRA"] = combined_data.groupby("trip_id")["ARR"].shift(1)
    combined_data["PRD"] = combined_data.groupby("trip_id")["DEP"].shift(1)

    # add a column for computing the arrival and departure delay at the previous 2 stops per entry
    combined_data["PRA_2"] = combined_data.groupby("trip_id")["ARR"].shift(2)
    combined_data["PRD_2"] = combined_data.groupby("trip_id")["DEP"].shift(2)
    
    
    combined_data["prev_DEP_time"] = combined_data.groupby("trip_id")["departure_time_sec"].shift(1)
    combined_data["travel_dur"] = combined_data["arrival_time_sec"] - combined_data["prev_DEP_time"]
    combined_data = combined_data.dropna()

    # get the other columns not present in the pivot table
    other_cols = data.drop(columns=["arrival_time_sec", "departure_time_sec"]) \
               .drop_duplicates(subset=["trip_id", "stop_sequence"])

    # merge those columns back into the df
    final_data = combined_data.merge(
        other_cols,
        on=["trip_id", "stop_sequence"],
        how="left"
    )

    # add a column that contains a binary value (0 or 1) if there is a delay at the current stop, the prev stop, or the prev 2 stops for classification purposes
    final_data["delayed"] = ((final_data["ARR"] > final_data["PRD"]) | (final_data["DEP"] > final_data["PRA"])).astype(int)
    final_data["prev_delayed"] = ((final_data["PRA"] > 60) | (final_data["PRD"] > 60)).astype(int)
    final_data["prev_delayed_2"] = ((final_data["PRA_2"] > 60) | (final_data["PRD_2"] > 60)).astype(int)

    final_data = final_data.fillna(0) 
    
    return final_data

def data_to_model(final_data):

    # add a column that contains a binary value (0 or 1) if there is a delay at the current stop, the prev stop, or the prev 2 stops for classification purposes
    final_data["delayed"] = ((final_data["ARR"] > 60) | (final_data["DEP"] > 60)).astype(int)
    final_data["prev_delayed"] = ((final_data["PRA"] > 60) | (final_data["PRD"] > 60)).astype(int)
    final_data["prev_delayed_2"] = ((final_data["PRA_2"] > 60) | (final_data["PRD_2"] > 60)).astype(int)

    final_data = final_data.fillna(0)
    final_data = final_data.sort_values(["trip_id", "stop_sequence"])
    
    # sort by time first
    final_data = final_data.sort_values("service_date")
    final_data.to_csv("help.csv")
    
    # define splits
    train_df = final_data[final_data["service_date"] < "2023-01-01"]
    val_df   = final_data[(final_data["service_date"] >= "2023-11-01") & 
                        (final_data["service_date"] < "2023-12-01")]
    test_df  = final_data[final_data["service_date"] >= "2023-12-01"]
    
    trip_labels = final_data.groupby("trip_id")["delayed"].max().reset_index()
    
    delayed_trips = trip_labels[trip_labels["delayed"] == 1]["trip_id"]
    not_delayed_trips = trip_labels[trip_labels["delayed"] == 0]["trip_id"]
    

    # downsample majority class
    min_size = min(len(delayed_trips), len(not_delayed_trips))

    delayed_sample = resample(delayed_trips, replace=False, n_samples=min_size, random_state=42)
    not_delayed_sample = resample(not_delayed_trips, replace=False, n_samples=min_size, random_state=42)

    balanced_trips = pd.concat([delayed_sample, not_delayed_sample])
    
    # filter df
    balanced_df = final_data[final_data["trip_id"].isin(balanced_trips)]
    
    trip_labels = balanced_df.groupby("trip_id")["delayed"].max().reset_index()
    
    trip_ids = trip_labels["trip_id"]
    y = trip_labels["delayed"]
    
    train_ids, temp_ids = train_test_split(
    trip_ids,
    test_size=0.3,
    stratify=y,
    random_state=42
    )

    temp_labels = trip_labels[trip_labels["trip_id"].isin(temp_ids)]

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.5,
        stratify=temp_labels["delayed"],
        random_state=42
    )
    
    train_df = balanced_df[balanced_df["trip_id"].isin(train_ids)]
    val_df   = balanced_df[balanced_df["trip_id"].isin(val_ids)]
    test_df  = balanced_df[balanced_df["trip_id"].isin(test_ids)]
    
    train_df.to_csv("train_df.csv")
    val_df.to_csv("val_df.csv")
    test_df.to_csv("test_df.csv")

    # fit seperately for train, test, validation set
    train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
    val_df[scale_cols] = scaler.transform(val_df[scale_cols])
    test_df[scale_cols] = scaler.transform(test_df[scale_cols])
    
    X_train, y_train, mask_train = create_trip_seqs(train_df)
    X_val, y_val, mask_val = create_trip_seqs(val_df)
    X_test, y_test, mask_test = create_trip_seqs(test_df)
    
    return X_train, y_train, mask_train, X_val, y_val, mask_val, X_test, y_test, mask_test
    
def create_trip_seqs(df):
    # need to send the data to the model in sequences (num_trips, max_seq_len, num_features)    
    # Group by trip_id and sort by stop_sequence
    trip_groups = df.groupby('trip_id')

    X_seqs = []
    y_seqs = []

    for trip_id, group in trip_groups:
        group = group.sort_values('stop_sequence')
        
        # Convert numerical features and target to tensors
        X = torch.tensor(group[features].values, dtype=torch.float32)
        y = torch.tensor(group[target].values, dtype=torch.float32)
        
        X_seqs.append(X)
        y_seqs.append(y)

    # Pad sequences to max length
    X_padded = pad_sequence(X_seqs, batch_first=True, padding_value=0.0)  # (batch, seq_len, features)
    # y_seqs: list of torch tensors with binary targets vals
    y_padded = pad_sequence(y_seqs, batch_first=True, padding_value=0.0)  # Use 0 for padding
    mask = pad_sequence([torch.ones(len(seq)) for seq in y_seqs], batch_first=True)  # 1 = valid and 0 = padded
    y_padded = y_padded.squeeze(-1)
    
    return X_padded, y_padded, mask

def train(X_train, y_train, mask_train, X_val, y_val, mask_val):
    input_size = X_train.shape[-1]
    hidden_size = 128
    model = DelayPredictor(input_size, hidden_size)

    # good for binary classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    dataset = TensorDataset(X_train, y_train, mask_train)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    count = 0
    prev_train_loss = 0
    epoch = 0
    wait = 15
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    for epoch in range(1000):
        model.train()
        total_loss = 0

        for X_batch, y_batch, mask_batch in loader:
            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            loss = (loss * mask_batch)
            loss = loss.sum() / (mask_batch.sum() + 1e-8)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(loader)
        val_loss = evaluate_model(model, X_val, y_val, mask_val, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "src/models/model_storage/best_model_2.pt")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= wait:
            print("Early stopping is triggered!")
            break

        print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")
        epoch += 1

    return model, train_losses, val_losses
        
def evaluate_model(model, X, y, mask, criterion):
    model.eval()
    
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
    
    return loss.item()

def compute_metrics(outputs, y_true, mask, threshold=0.5):
    """
    Compute masked accuracy, precision, and recall for binary predictions.
    """
    
    # selects only valid time steps (filters out padding)
    masked_preds = outputs[mask == 1]
    masked_actual = y_true[mask == 1]
    
    cm = confusion_matrix(masked_actual, masked_preds)
    tn, fp, fn, tp = cm.ravel() #retrieve the true negative, false positive, false negative, and true positive values from the matrix
    
    accuracy = (masked_preds == masked_actual).float().mean().item()
    
    # Class 1 (delayed)
    tp1 = ((masked_preds == 1) & (masked_actual == 1)).sum()
    fp1 = ((masked_preds == 1) & (masked_actual == 0)).sum()
    fn1 = ((masked_preds == 0) & (masked_actual == 1)).sum()
    precision_delayed = tp / (tp + fp + 1e-8)
    recall_delayed = tp / (tp + fn + 1e-8)
    
    f1_delayed = (2 * precision_delayed * recall_delayed) / (precision_delayed + recall_delayed + 1e-8)

    # Class 0 (not delayed)
    tp0 = ((masked_preds == 0) & (masked_actual == 0)).sum()
    fp0 = ((masked_preds == 0) & (masked_actual == 1)).sum()
    fn0 = ((masked_preds == 1) & (masked_actual == 0)).sum()
    precision_not_delayed = tn / (tn + fn + 1e-8)
    recall_not_delayed = tn / (tn + fp + 1e-8)
    
    f1_not_delayed = 2 * (precision_not_delayed * recall_not_delayed) / (precision_not_delayed + recall_not_delayed)
    
    return accuracy, precision_delayed, recall_delayed, f1_delayed, precision_not_delayed, recall_not_delayed, f1_not_delayed, (tn, fp, fn, tp)
    
def visualize_training_progress(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig("src/visualizations/lstm-train-2.png")
    
    
def main(is_train = True):
    final_data = preprocess_for_model()
    X_train, y_train, mask_train, X_val, y_val, mask_val, X_test, y_test, mask_test = data_to_model(final_data)
    
    if is_train:
        model, train_losses, val_losses = train(X_train, y_train, mask_train, X_val, y_val, mask_val)
        visualize_training_progress(train_losses, val_losses)
    else:
        test(X_test, y_test, mask_test)
    
def test(X_test, y_test, mask_test):
    model = DelayPredictor(input_size=8, hidden_size=128)

    # load weights
    model.load_state_dict(torch.load("src/models/model_storage/lstm_model.pt"))

    # eval mode
    model.eval()

    # run test batch
    with torch.no_grad():
        # send test data to the model
        outputs = model(X_test)  
        
        # converting the y_test and output tensors to numpy arrays to calculate the evaluation metrics 
        y_true = y_test[mask_test == 1].cpu().numpy().astype(int)
        y_pred = torch.sigmoid(outputs)[mask_test == 1].cpu().numpy()

        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

        f1 = (2 * precision * recall) / (precision + recall + 1e-8)
        
        f1 = f1[:-1]  # thresholds is shorter by 1

        best_idx = np.argmax(f1) 
        best_thresh = thresholds[best_idx]

        print("Best threshold:", best_thresh)
        print("Best F1:", f1[best_idx])
        print("Best Precision:", precision[best_idx])
        print("Best Recall:", recall[best_idx])
        

        # apply best threshold on predictions
        y_preds = (y_pred >= best_thresh).astype(int)

        # obtain confusion matrix
        cm = confusion_matrix(y_true, y_preds)
        tn, fp, fn, tp = cm.ravel()

        print("\nConfusion Matrix:")
        print(cm)

        print("\nTN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)
            
class ReadyLSTM:
    def __init__(self):
        self.df = preprocess_for_model()
        self.lookup = self.build_delay_lookup()
        data_to_model(self.df)
        self.scaler = scaler
    
    def build_delay_lookup(self):
        grouped = self.df.groupby(
            ["route_id", "stop_name", "hour", "day_of_week_num"]
        )["delay_sec"].mean()

        return grouped.to_dict()
     
    def get_valid_trips(self, route_df, origin_stop, destination_stop):
        """    
        returns existing trips that go from the origin stop to the destination stop      
        """
        trips = []
        trips_in_route = route_df.groupby("trip_id")

        for trip_id, trip in trips_in_route:
            trip = trip.sort_values("stop_sequence")

            stops = trip["stop_name"].values

            if origin_stop in stops and destination_stop in stops:
                origin_idx = trip[trip["stop_name"] == origin_stop]["stop_sequence"].values[0]
                dest_idx = trip[trip["stop_name"] == destination_stop]["stop_sequence"].values[0]

                # checks if the origin stop is before the dest stop in this trip (going in the right dir)
                if origin_idx < dest_idx:
                    trips.append(trip_id)

        return trips
    
    def get_next_trip(self, route_df, trips, origin_stop, current_time_sec):
        """    
        get next possible arrival time at arrival stop and "trip" the user wants to go on
        """
        candidates = []
        for trip_id in trips:
            trip = route_df[route_df["trip_id"] == trip_id]

            origin_row = trip[trip["stop_name"] == origin_stop]
            origin_time = origin_row["arrival_time_sec"].values[0]

            if origin_time >= current_time_sec:
                candidates.append((trip_id, origin_time))

        if not candidates:
            return None

        # earliest future trip
        return sorted(candidates, key=lambda x: x[1])[0][0]
    
    def build_trip_sequence(self, route_df, trip_id, origin_stop, destination_stop):
        """    
        returns the entire expected route arr/dest info from origin -> destination 
        """
        trip = route_df[route_df["trip_id"] == trip_id].copy()
        # so the stops are in order
        trip = trip.sort_values("stop_sequence")

        origin_seq = trip[trip["stop_name"] == origin_stop]["stop_sequence"].values[0]
        dest_seq = trip[trip["stop_name"] == destination_stop]["stop_sequence"].values[0]

        seq = trip[(trip["stop_sequence"] >= origin_seq) &
                (trip["stop_sequence"] <= dest_seq)].copy()

        return seq
    
    def add_time_features(self, seq_df, current_dt):
        seq_df = seq_df.copy()

        seq_df["hour"] = current_dt.hour
        seq_df["day_of_week_num"] = current_dt.weekday()
        seq_df["month_num"] = current_dt.month

        return seq_df

    def one_hot_encode_route(self, seq_df, route_cols, route):
        """    
        one hot encoding of the route
        """
        for col in route_cols:
            if col not in seq_df.columns:
                if route in col:
                    seq_df[col] = 1
                else:
                    seq_df[col] = 0

        return seq_df

    def estimate_prev_delayed(self, seq_df, expected_delay_dict, route_id):
        """    
        use existing knowledge of delay patterns for a specific route, stop, hour
        """
        
        prev = 0
        prev2 = 0

        prev_list = []
        prev2_list = []

        for _, row in seq_df.iterrows():
            key = (
                route_id,
                row["stop_name"],
                row["hour"],
                row["day_of_week_num"]
            )

            exp_delay = expected_delay_dict.get(key, 0.1)

            prev_list.append(prev)
            prev2_list.append(prev2)

            prev2 = prev
            prev = exp_delay

        seq_df["prev_delayed"] = prev_list
        seq_df["prev_delayed_2"] = prev2_list

        return seq_df

    def to_tensor(self, seq_df):
        X = seq_df[features].values
        tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        return tensor
    
    def time_to_seconds(self, dt):
        return dt.hour * 3600 + dt.minute * 60 + dt.second
    
    def predict_prob(self, model, X_tensor):
        model.eval()
        with torch.no_grad():
            out = model(X_tensor)
            probs = torch.sigmoid(out)

        return probs.squeeze().numpy()
    
    def predict(self, obs):
        route, arr_stop, dest_stop, cur_datetime = obs
        current_time_sec = self.time_to_seconds(cur_datetime)
        
        expected_delay_dict = {}

        grouped = self.df.groupby(
            ["route_id", "stop_name", "hour", "day_of_week_num"]
        )["delay_sec"].mean()
        expected_delay_dict = grouped.to_dict()
        
        # filtering df to only include entries of a specific route 
        route_df = self.df[self.df["route_id"] == route]
        trips = self.get_valid_trips(route_df, arr_stop, dest_stop)
        next_trip = self.get_next_trip(route_df, trips, arr_stop, current_time_sec)
        trip_seq_df_1 = self.build_trip_sequence(route_df, next_trip, arr_stop, dest_stop)
        trip_seq_df_2 = self.add_time_features(trip_seq_df_1, cur_datetime)
        trip_seq_df_3 = self.one_hot_encode_route(trip_seq_df_2, features[12:], route)
        final_seq_input = self.estimate_prev_delayed(trip_seq_df_3, expected_delay_dict, route)
        filtered_input = final_seq_input[features]
        
        filtered_input[scale_cols] = self.scaler.transform(filtered_input[scale_cols])
        
        X = self.to_tensor(filtered_input)

        
        model = DelayPredictor(input_size=8, hidden_size=128)

        # load weights of the best model
        model.load_state_dict(torch.load("src/models/model_storage/lstm_model.pt"))

        probs = self.predict_prob(model, X)

        arrival_prob = probs[0]
        destination_prob = probs[-1]
        delayed = False
        if destination_prob > 0.7 and arrival_prob > 0.7:
            delayed = "Very Likely"
        if destination_prob > 0.5 and arrival_prob > 0.5:
            delayed = "Likely"
        else:
            delayed = "Not Likely"
            
        return trip_seq_df_1, {
            "Arrival at Starting Point Delay": f'The LSTM predicts the next {route} Line train arriving at {arr_stop} has a {int(arrival_prob * 100)}% chance of being delayed',
            "Arrival at Destination Delay": f'The LSTM predicts the next {route} Line train heading to {dest_stop} has a {int(destination_prob * 100)}% chance of being delayed',
            "Delay is": delayed
        }
        
        
# testing the model with mock user input before frontend was finalized
# if __name__ == "__main__":

#     final_data = preprocess_for_model()

#     obs = ["Orange", "Downtown Crossing", "Massachusetts Avenue",
#            pd.Timestamp(2026, 4, 9, 6, 30)]

#     print(predict(obs))
  
# to train  
# main(True)

# to test
# main(Test)
    
    
    

    