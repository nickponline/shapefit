## 500 unseen scenes, all families (scene_set(500, seed=999))
| method | IoU ↑ | vertex count exact ↑ | vertex F1 @3% ↑ | vertex err ↓ | Hausdorff ↓ |
|---|---|---|---|---|---|
| shapefit (NN) | 0.960 | 78.8% | 0.845 | 1.32% | 2.14% |
| shapefit (NN) + TTA | 0.960 | 77.0% | 0.838 | 1.29% | 2.21% |
| approxPolyDP 1% | 0.910 | 23.6% | 0.541 | 2.71% | 5.76% |
| approxPolyDP 2% | 0.900 | 47.8% | 0.617 | 2.71% | 5.73% |
| approxPolyDP 3% | 0.876 | 50.6% | 0.602 | 2.58% | 7.39% |
| convex hull + DP | 0.891 | 24.6% | 0.581 | 2.81% | 5.56% |
| min-area rectangle | 0.733 | 36.8% | 0.197 | 14.13% | 22.45% |

## 200 unseen unions of overlapping rectangles (scene_set(200, seed=1000, family='rect_union'))
| method | IoU ↑ | vertex count exact ↑ | vertex F1 @3% ↑ | vertex err ↓ | Hausdorff ↓ |
|---|---|---|---|---|---|
| shapefit (NN) | 0.947 | 67.0% | 0.818 | 1.39% | 3.10% |
| shapefit (NN) + TTA | 0.949 | 68.5% | 0.829 | 1.29% | 2.78% |
| approxPolyDP 1% | 0.898 | 28.0% | 0.588 | 2.64% | 5.80% |
| approxPolyDP 2% | 0.880 | 25.5% | 0.578 | 2.67% | 7.08% |
| approxPolyDP 3% | 0.844 | 13.0% | 0.519 | 2.56% | 9.99% |
| convex hull + DP | 0.825 | 9.0% | 0.513 | 4.69% | 15.61% |
| min-area rectangle | 0.690 | 8.0% | 0.106 | 14.34% | 24.94% |

## per family, 500 unseen scenes
| family | n | shapefit IoU | shapefit count exact | shapefit F1 | DP 2% IoU | DP 2% count exact | DP 2% F1 |
|---|---|---|---|---|---|---|---|
| `convex` | 109 | 0.965 | 56.9% | 0.732 | 0.914 | 24.8% | 0.586 |
| `ngon` | 92 | 0.965 | 94.6% | 0.903 | 0.917 | 70.7% | 0.634 |
| `quad` | 65 | 0.951 | 87.7% | 0.828 | 0.891 | 63.1% | 0.641 |
| `rect_union` | 29 | 0.940 | 62.1% | 0.817 | 0.850 | 10.3% | 0.478 |
| `rectangle` | 78 | 0.972 | 93.6% | 0.956 | 0.906 | 71.8% | 0.685 |
| `rectilinear` | 49 | 0.965 | 87.8% | 0.914 | 0.911 | 30.6% | 0.610 |
| `star` | 78 | 0.948 | 69.2% | 0.804 | 0.874 | 41.0% | 0.611 |
