## memorised scenes (training set of the overfit run)
| method | IoU ↑ | vertex count exact ↑ | vertex F1 @3% ↑ | vertex err ↓ | Hausdorff ↓ |
|---|---|---|---|---|---|
| shapefit (NN) | 0.994 | 100.0% | 0.997 | 0.19% | 0.29% |
| shapefit (NN) + TTA | 0.942 | 8.3% | 0.559 | 2.19% | 3.45% |
| approxPolyDP 1% | 0.924 | 29.2% | 0.558 | 2.25% | 4.91% |
| approxPolyDP 2% | 0.923 | 64.6% | 0.641 | 2.43% | 4.31% |
| approxPolyDP 3% | 0.903 | 64.6% | 0.648 | 2.43% | 5.59% |
| convex hull + DP | 0.906 | 25.0% | 0.582 | 2.50% | 5.04% |
| min-area rectangle | 0.747 | 43.8% | 0.216 | 11.71% | 20.53% |

## 200 unseen scenes (overfit model; expected to be poor)
| method | IoU ↑ | vertex count exact ↑ | vertex F1 @3% ↑ | vertex err ↓ | Hausdorff ↓ |
|---|---|---|---|---|---|
| shapefit (NN) | 0.923 | 0.5% | 0.344 | 3.07% | 5.30% |
| shapefit (NN) + TTA | 0.932 | 3.5% | 0.452 | 2.83% | 4.51% |
| approxPolyDP 1% | 0.911 | 29.0% | 0.552 | 2.66% | 5.52% |
| approxPolyDP 2% | 0.900 | 46.0% | 0.608 | 2.48% | 5.67% |
| approxPolyDP 3% | 0.875 | 47.0% | 0.595 | 2.65% | 7.79% |
| convex hull + DP | 0.888 | 29.0% | 0.583 | 2.82% | 5.78% |
| min-area rectangle | 0.718 | 36.0% | 0.193 | 15.07% | 23.60% |
