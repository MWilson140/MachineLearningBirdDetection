import torch


class ModelTester:
    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device

    def test_model(self):
        correct = 0
        total_samples = 0
        true_positive = 0  # Yes correct
        true_negative = 0  # No correct
        false_positive = 0 # Yes incorrect
        false_negative = 0 # No incorrect

        # Counters for totals and cumulative confidence
        yes_count = 0
        no_count = 0
        yes_confidence_sum = 0.0
        no_confidence_sum = 0.0

        self.model.eval()
        with torch.no_grad():
            print(f"{'File ID':<25}{'Prediction':<15}{'Actual':<15}{'Confidence (%)':<20}{'Correct':<10}")
            print("=" * 80)
            for i, data in enumerate(self.data_loader):
                inputs, target, file_id = data
                inputs, target = inputs.to(self.device), target.to(self.device)
                outputs = self.model(inputs)
                probability = torch.sigmoid(outputs)
                predictions = (probability > 0.5).int().squeeze()

                correct += (predictions == target).sum().item()
                total_samples += target.size(0)

                for j in range(len(predictions)):
                    predicted_label = predictions[j].item()
                    actual_label = target[j].item()
                    current_file_id = file_id[j]

                    if predicted_label == 1 and actual_label == 1:
                        true_positive += 1  # Yes correct
                    elif predicted_label == 0 and actual_label == 0:
                        true_negative += 1  # No correct
                    elif predicted_label == 1 and actual_label == 0:
                        false_positive += 1  # Yes incorrect
                    elif predicted_label == 0 and actual_label == 1:
                        false_negative += 1  # No incorrect

                    predicted_str = "yes" if predicted_label == 1 else "no"
                    actual_str = "yes" if actual_label == 1 else "no"
                    prediction_percentage = probability[j].item() * 100
                    is_correct = "True" if predicted_label == actual_label else "False"

                    # Track confidence based on the ground truth
                    if actual_label == 1:  # Ground truth is "yes"
                        yes_count += 1
                        yes_confidence_sum += prediction_percentage
                    else:  # Ground truth is "no"
                        no_count += 1
                        no_confidence_sum += prediction_percentage

                    print(f"{current_file_id:<25}{predicted_str:<15}{actual_str:<15}{prediction_percentage:<20.2f}{is_correct:<10}")

        # Final accuracy calculation
        accuracy = correct / total_samples * 100
        print("=" * 80)
        print(f"Test Accuracy: {accuracy:.2f}%")
        print(f"Yes Correct (True Positive): {true_positive}")
        print(f"No Correct (True Negative): {true_negative}")
        print(f"Yes Incorrect (False Positive): {false_positive}")
        print(f"No Incorrect (False Negative): {false_negative}")

        # Print average confidence for each ground truth label
        print("\nAverage Confidence for Each Ground Truth Label:")
        print(f"Yes (Ground Truth = 1): {yes_confidence_sum / yes_count:.2f}%" if yes_count > 0 else "No 'yes' samples")
        print(f"No (Ground Truth = 0): {no_confidence_sum / no_count:.2f}%" if no_count > 0 else "No 'no' samples")