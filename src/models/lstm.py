# LSTM
# improved version of RNN to capture long term dependencies in sequential data 
# good for time series forecasting
# LSTM architecture: involves what goes/stays in the memory cell
    # Input Gate: controls what information is added to the memory cell
        # equation: sigmoid(W_i * [h_t-1, x_t] + b_i) # information is regulated using the sigmoid function and filter the values to be remembered
        # C_t = tanh(W_c * [h_t-1, x_t] + b_c) # a vector is created using tanh function that gives an output from -1 to +1 which contains all the possible values from (??? this may mean each element) prev hidden state to cur input
        # these are multiplied tg to get regulated inputs * vector values to add useful information to the memory cell
        # multiply f_t (forget gate output) by the previous C_t to filter out the information we decided to ignore previously (and) and add that with the new multiplication result (or)
    # Forget Gate: determines what information is removed from the memory cell
        # equation: sigmoid(W_f * [h_t-1, x_t] + b_f) 
        # W_f = weight matrix for forget (f) gate
        # h_t-1 = previous hidden state
        # x_t = current input state
        # b_f = bias factor for bias (b) gate
    # Output Gate: decides what part of the current memory cell state should be sent as the hidden state for a given time step
        # equation: sigmoid(W_o * [h_t-1, x_t] + b_o) # which information from the current cell state will be output
        # h_t = o_t * tanh(C_t) # current cell state is passed to a tanh function that gives an output from -1 to +1 and multiplied by the sigmoid equation output --> hidden state h_t
    # Output Gate: controls what memory information is outputted from the memory cell
# the outputted C_t (current cell state) from the forget and input gate and h_t from the output gate are used to generate the next output of the network!
# This allows LSTM networks to selectively retain or discard information as it flows through the network to learn long-term dependencies
# hidden state is treated as its "short term memory"
# the memory is updated using the current input, the previous hidden state and the current state of the memory cell.
    # memory = current input + prev hidden state (short term memory) + cur state memory cell
    # h_t
    
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # weights for all gates combined
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)
    
    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        # device = x.device #???
        
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
    
    # predicting delay at each stop
    def forward(self, x):
        outputs, _ = self.lstm(x) # (batch, seq_len, hidden_size)
        predictions = torch.sigmoid(self.fc(outputs)) # (batch, seq_len, 1)
        return predictions.squeeze(-1) # (batch, seq_len)
    
    
 
# TODO: initialize vars somewhere else
day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}  

features = ['stop_sequence', 'ARR', 'DEP', 'PRA', 'PRD', 'PRA_2', 'PRD_2', 'day_of_week_num', 'prev_delayed', 'prev_delayed_2', 'month_num', 'hour']
target = ['delayed']
delay_val_cols = ['ARR', 'DEP', 'PRA', 'PRD', 'PRA_2', 'PRD_2']

 
def preprocess_for_model():
    data = pd.read_csv('Datasets/final_data.csv')
    # convert day of the week and month to their numerical representations to send to the model
    data["day_of_week_num"] = data["day_of_week"].map(day_map)
    data['month_num'] = data['service_date'].str[5:7].astype(int)
    
    # combined the arrival and departure delay into one row per trip id and stop id
    combined_data = data.pivot_table(
        index=["trip_id", "stop_sequence"],
        columns="event_type",
        values="delay_sec"
    ).reset_index()

    # add a column for computing the arrival and departure delay at the previous stop per entry (delays usually have a domino effect)
    combined_data["PRA"] = combined_data.groupby("trip_id")["ARR"].shift(1)
    combined_data["PRD"] = combined_data.groupby("trip_id")["DEP"].shift(1)

    # add a column for computing the arrival and departure delay at the previous 2 stops per entry
    combined_data["PRA_2"] = combined_data.groupby("trip_id")["ARR"].shift(2)
    combined_data["PRD_2"] = combined_data.groupby("trip_id")["DEP"].shift(2)

    # get the other columns not present in the pivot table
    other_cols = data.drop_duplicates(subset=["trip_id", "stop_sequence"])

    # merge those columns back into the df
    final_data = combined_data.merge(
        other_cols,
        on=["trip_id", "stop_sequence"],
        how="left"
    )

    # add a column that contains a binary value (0 or 1) if there is a delay at the current stop, the prev stop, or the prev 2 stops for classification purposes
    final_data["delayed"] = ((final_data["ARR"] > 60) | (final_data["DEP"] > 60)).astype(int)
    print(final_data["delayed"].value_counts())
    final_data["prev_delayed"] = ((final_data["PRA"] > 60) | (final_data["PRD"] > 60)).astype(int)
    final_data["prev_delayed_2"] = ((final_data["PRA_2"] > 60) | (final_data["PRD_2"] > 60)).astype(int)

    final_data = final_data.fillna(0)
    final_data = final_data.sort_values(["trip_id", "stop_sequence"])
    
    # scale the delay values so they are all centered around a specific point and are comparable without some delay values outweighing others (?)
    delay_val_scaler = StandardScaler()
    final_data[delay_val_cols] = delay_val_scaler.fit_transform(final_data[delay_val_cols])
    
    # need to send the data to the model in sequences (num_trips, max_seq_len, num_features)    
    # Group by trip_id and sort by stop_sequence
    trip_groups = final_data.groupby('trip_id')

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


    print("X shape:", X_padded.shape)
    print("y shape:", y_padded.shape)
    print("mask shape:", mask.shape)
    
    return X_padded, y_padded, mask

def train(X_train, y_train, mask_train, X_val, y_val, mask_val):
    input_size = X_train.shape[-1]
    hidden_size = 128
    model = DelayPredictor(input_size, hidden_size)

    # good for binary classification
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dataset = TensorDataset(X_train, y_train, mask_train)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    count = 0
    prev_train_loss = 0
    epoch = 0
    while count < 5:
        model.train()
        total_loss = 0
        best_val_loss = float('inf')

        for X_batch, y_batch, mask_batch in loader:
            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            loss = (loss * mask_batch).sum() / (mask_batch.sum() + 1e-8)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(loader)
        if train_loss == prev_train_loss:
            count += 1
        else:
            count = 0
        prev_train_loss = train_loss

        val_loss = evaluate_model(model, X_val, y_val, mask_val, criterion)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/model_storage/best_model.pt")

        print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")
        epoch += 1

    return model
        
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
    
    outputs: (batch, seq_len) raw probabilities from DelayPredictor
    y_true: (batch, seq_len) ground truth 0/1
    mask: (batch, seq_len) 0/1 mask indicating valid timesteps
    threshold: probability cutoff to predict 1 (delayed)
    """
    
    # selects only valid time steps (filters out padding)
    masked_preds = outputs[mask == 1]
    masked_actual = y_true[mask == 1]
    
    accuracy = (masked_preds == masked_actual).float().mean().item()
    
    # Class 1 (delayed)
    tp1 = ((masked_preds == 1) & (masked_actual == 1)).sum()
    fp1 = ((masked_preds == 1) & (masked_actual == 0)).sum()
    fn1 = ((masked_preds == 0) & (masked_actual == 1)).sum()
    precision1 = tp1 / (tp1 + fp1 + 1e-8)
    recall1 = tp1 / (tp1 + fn1 + 1e-8)

    # Class 0 (not delayed)
    tp0 = ((masked_preds == 0) & (masked_actual == 0)).sum()
    fp0 = ((masked_preds == 0) & (masked_actual == 1)).sum()
    fn0 = ((masked_preds == 1) & (masked_actual == 0)).sum()
    precision0 = tp0 / (tp0 + fp0 + 1e-8)
    recall0 = tp0 / (tp0 + fn0 + 1e-8)
    
    # for durations: simple mean squared errors
    # for classifications: accuracy / precisions chart TP, FN, FP, TN
    # penalize for how high or how low classification is 
    # mse but values are 0 to 1 (maybe...)
    
    return accuracy, precision1, recall1, precision0, recall0
    
def main(is_train = True):
    X_padded, y_padded, mask = preprocess_for_model()
    
    # Step 1: Train + temp (val+test)
    X_train, X_temp, y_train, y_temp, mask_train, mask_temp = train_test_split(
        X_padded, y_padded, mask, test_size=0.3, random_state=42
    )

    # Step 2: Split temp into validation and test (50/50 of temp = 15% val, 15% test)
    X_val, X_test, y_val, y_test, mask_val, mask_test = train_test_split(
        X_temp, y_temp, mask_temp, test_size=0.5, random_state=42
    )
    if is_train:
        model = train(X_train, y_train, mask_train, X_val, y_val, mask_val)
    else:
        test(X_test, y_test, mask_test)
    
def test(X_test, y_test, mask_test):
    model = DelayPredictor(input_size=12, hidden_size=128)

    # load weights
    model.load_state_dict(torch.load("models/model_storage/best_model.pt"))

    # eval mode
    model.eval()

    # run test batch
    with torch.no_grad():
        # send test data to the model
        outputs = model(X_test)  
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            # Convert probabilities to binary predictions
            predictions = (outputs > threshold).float()
            accuracy, precision1, recall1, precision0, recall0 = compute_metrics(predictions, y_test, mask_test)

            print(f"Threshold: {threshold}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision (predicting delayed): {precision1:.4f}")
            print(f"Recall (predicting delayed): {recall1:.4f}") 
            print(f"Precision (predicting not delayed): {precision0:.4f}")
            print(f"Recall (predicting not delayed): {recall0:.4f}")
        
    
main(False)
    
    
    

    