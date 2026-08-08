

<!-- Start of picture text -->
nailergla ee Bateesee ge ia or oe<br>ar age onthe canteens ntl “he elo SE aimee:<br>’ “ilrid| r ~- a<\ve tl \ .<br>> cull zALT 0 es a<br><!-- End of picture text -->













<!-- Start of picture text -->
Depth Grid Association Feat. Aggregation<br>Index 0 0 1 1 1 2 2 2<br>0.1 0.1 0.1 Value 1 3 7 -1 -2 4 -3 6<br>0.50.2 0.1 0.1 Prefix Sum Reduction ( LSS ) Precomputation: Interval Reduction:  1.9 × 22.1 ×<br>0.2 0.50.3 0.60.2 Pref. SumResults 1 4 4 11 10 8 4 12 9 15 7 0 (c) Improvement breakdown20 40 log scale 500ms<br>B<br>Interval Reduction ( Ours )<br>C LSS: 512.1ms<br>A Thread 1 Thread 2 Thread 3 136.8ms 2127.3ms<br>Results 4 4 7 Ours:4.8ms 12.0ms 45.1ms<br>: Stored to DRAM ( Slow )<br>(a) Camera-to-BEV  1/16 FPN 1/8 FPN 1/4 FPN<br>transformation (b) Efficient BEV pooling (d) Scalability<br><!-- End of picture text -->

Fig. 3: Camera-to-BEV transformation (a) is the key step to perform sensor fusion in the unified BEV space. Existing implementation is extremely slow and takes up to 2s for a single scene. We propose efficient BEV pooling (b) using interval reduction and fast grid association with precomputation, bringing about **40** _×_ speedup to view transformation (c, d). 

LiDAR/radar features are typically in the 3D/bird’s-eye view. Even for camera features, each one of them has a distinct viewing angle ( _i.e_ ., front, back, left, right). This _view discrepancy_ makes the feature fusion difficult since the same element in different feature tensors might correspond to very different spatial locations (and the na¨ıve elementwise feature fusion will not work in this case). Thus, it is crucial to find a _shared_ representation, such that (1) all sensor features can be easily converted to it without information loss, and (2) it is suitable for different types of tasks. 

**To Camera.** Motivated by RGB-D data, one choice is to project the LiDAR point cloud to the camera plane and render the 2.5D sparse depth. However, this conversion is _geometrically lossy_ . Two neighbors on the depth map can be far away from each other in the 3D space. This makes the camera view less effective for tasks that focus on the object/scene geometry, such as 3D object detection. 

**To LiDAR.** Most state-of-the-art sensor fusion methods [1], [4], [3] decorate LiDAR points with their corresponding camera features ( _e.g_ ., semantic labels, CNN features or virtual points). However, this camera-to-LiDAR projection is _semantically lossy_ . Camera and LiDAR features have drastically different densities, resulting in only less than 5% of camera features being matched to a LiDAR point (for a 32-channel LiDAR scanner). Giving up the semantic density of camera features severely hurts the model’s performance on semantic-oriented tasks (such as BEV map segmentation). Similar drawbacks also apply to more recent fusion methods in the latent space ( _e.g_ ., object query) [48], [49]. 

**To Bird’s-Eye View.** We _adopt the bird’s-eye view (BEV) as the unified representation for fusion_ . This view is friendly to almost all perception tasks since the output space is also in BEV. More importantly, the transformation to BEV keeps both geometric structure (from LiDAR features) and semantic density (from camera features). On the one hand, the LiDARto-BEV projection flattens the sparse LiDAR features along the height dimension, thus does not create geometric distortion in Figure 1a. On the other hand, camera-to-BEV projection casts each camera feature pixel back into a ray in the 3D space (detailed in the next section), which can result in a 

dense BEV feature map in Figure 1c that retains full semantic information from the cameras. 

# _B. Efficient Camera-to-BEV Transformation_ 

Camera-to-BEV transformation is non-trivial because the depth associated with each camera feature pixel is inherently ambiguous. Following LSS [6], we explicitly predict the discrete depth distribution of each pixel. We then scatter each feature pixel into _D_ discrete points along the camera ray and rescale the associated features by their corresponding depth probabilities (Figure 3a). This generates a camera feature point cloud of size _NHWD_ , where _N_ is the number of cameras and ( _H, W_ ) is the camera feature map size. Such 3D feature point cloud is quantized along the _x, y_ axes with a step size of _r_ ( _e.g_ ., 0.4m). We use the _BEV pooling_ operation to aggregate all features within each _r × r_ BEV grid and flatten the features along the _z_ -axis. 

Though simple, BEV pooling is surprisingly inefficient and slow, taking more than 500ms on an RTX 3090 GPU (while the rest of our model only takes around 100ms). This is because the camera feature point cloud is very large: for a typical workload<sup>*</sup> , there could be around 2 million points generated for each frame, two orders of magnitudes denser than a LiDAR feature point cloud. To lift this efficiency bottleneck, we propose to optimize the BEV pooling with precomputation and interval reduction. 

**Precomputation.** The first step of BEV pooling is to _associate_ each point in the camera feature point cloud with a BEV grid. Different from LiDAR point clouds, the coordinates of the camera feature point cloud are _fixed_ (as long as the camera intrinsics and extrinsics stay the same, which is usually the case after proper calibration). Motivated by this, we precompute the 3D coordinate and the BEV grid index of each point. We also sort all points according to grid indices and record the rank of each point. During inference, we only need to reorder all feature points based on the precomputed ranks. This caching mechanism can reduce the latency of grid association from 17ms to 4ms. 

> * _N_ = 6, ( _H, W_ ) = (32 _,_ 88), and _D_ = (60 _−_ 1) _/_ 0 _._ 5 = 118. This corresponds to six multi-view cameras, each associated with a 32 _×_ 88 camera feature map (which is downsampled from a 256 _×_ 704 image by 8 _×_ ). The depth is discretized into [1 _,_ 60] meters with a step size of 0.5 meter. 

Tab. I: BEVFusion achieves state-of-the-art 3D object detection performance on nuScenes (val and test) without bells and whistles. It breaks the convention of decorating camera features onto the LiDAR point cloud and delivers at least 1.3% higher mAP and NDS with **1.5-2** _×_ lower computation cost. (<sup>_∗_</sup> : our re-implementation;<sup>_†_</sup> : with test-time augmentation) 

||Modality|mAP (_test_)|NDS (_test_)|mAP (_val_)|NDS (_val_)|MACs (G)|Latency (ms)|
|---|---|---|---|---|---|---|---|
|M<sup>2</sup>BEV [41]|C|42.9|47.4|41.7|47.0|–|–|
|BEVFormer [43]|C|44.5|53.5|41.6|51.7|–|–|
|PointPillars [10]|L|–|–|52.3|61.3|65.5|34.4|
|SECOND [11]|L|52.8|63.3|52.6|63.0|85.0|69.8|
|CenterPoint [17]|L|60.3|67.3|59.6|66.8|153.5|80.7|
|PointPainting [1]|C+L|–<br>|–<br>|65.8<sup>_∗_</sup>|69.6<sup>_∗_</sup>|370.0|185.8|
|PointAugmenting [2]|C+L|66.8<sup>_†_</sup>|71.0<sup>_†_</sup>|–|–|408.5|234.4|
|MVP [4]|C+L|66.4|70.5|66.1<sup>_∗_</sup>|70.0<sup>_∗_</sup>|371.7|187.1|
|FusionPainting [50]|C+L|68.1|71.6|66.5|70.7|–|–|
|AutoAlign [51]|C+L|–|–|66.6|71.1|–|–|
|FUTR3D [48]|C+L|–|–|64.5|68.3|1069.0|321.4|
|TransFusion [49]|C+L|68.9|71.6|67.5|71.3|485.8|156.6|
|**BEVFusion** (Ours)|C+L|**70.2**|**72.9**|**68.5**|**71.4**|**253.2**|**119.2**|



**Interval Reduction.** After grid association, all points within the same BEV grid will be consecutive in the tensor representation. The next step of BEV pooling is to _aggregate_ the features within each BEV grid by some symmetric function ( _e.g_ ., mean, max, and sum). As in Figure 3b, existing implementation [6] first computes the prefix sum over all points and then subtracts the values at the boundaries where indices change. However, the prefix sum operation requires tree reduction on the GPU and produces many unused partial sums (since we only need those values on the boundaries), both of which are inefficient. To accelerate feature aggregation, we implement a specialized GPU kernel that parallelizes directly over BEV grids: we assign a GPU thread to each grid that calculates its interval sum and writes the result back. This kernel removes the dependency between outputs (thus does not require multi-level tree reduction) and avoids writing the partial sums to the DRAM, reducing the latency of feature aggregation from 500ms to 2ms (Figure 3c). 

**Takeaways.** The camera-to-BEV transformation is **40** _×_ faster with our optimized BEV pooling: the latency is reduced from more than 500ms to 12ms (only 10% of our model’s end-to-end runtime) and scales well across different feature resolutions (Figure 3d). This is a key enabler for unifying multi-modal sensory features in the shared BEV representation. Two concurrent works of ours also identify this efficiency bottleneck in the camera-only 3D detection. They approximate the view transformer by assuming uniform depth distribution [41] or truncating the points within each BEV grid [40]. In contrast, our techniques are _exact_ without any approximation, while still being faster. 

Tab. II: BEVFusion achieves state-of-the-art 3D object detection performance among all submissions on Waymo open dataset (test). (<sup>_†_</sup> : with test-time augmentation,<sup>_‡_</sup> : with both test-time augmentation and model ensemble) 

||Frames|mAP/L1|mAPH/L1|mAP/L2|mAPH/L2|
|---|---|---|---|---|---|
|AFDetV2-Ens [18]_‡_|**3**|84.1|82.6|79.0|77.6|
|InceptionLiDAR|10|83.8|82.5|79.2|77.8|
|3DAL-Ens [20]|5|84.6|83.1|79.7|78.2|
|DeepFusion-Ens [3]_‡_|5|84.4|83.2|79.5|78.4|
|MT-Net_‡_ [55]|3|84.7|83.2|79.9|78.5|
|MT3D|4|85.0|83.7|80.1|78.7|
|LIVOX-Detection|7|84.8|83.5|80.2|79.0|
|MPPNet-Ens_‡_ [56]|16|85.0|83.7|80.5|79.1|
|3DAM-Ens|5|85.3|83.8|80.7|79.2|
|**BEVFusion** (Ours)_†_|**3**|**85.7**|**84.4**|**80.8**|**79.5**|



blocks) to compensate for such local misalignments. Our method could potentially benefit from more accurate depth estimation ( _e.g_ ., supervising the view transformer with groundtruth depth [42], [53]), which we leave for future work. 

# _D. Multi-Task Heads_ 

We apply multiple task-specific heads to the fused BEV feature map. Our method is applicable to most 3D perception tasks. For 3D object detection, we follow [17], [49] to use a class-specific center heatmap head to predict the center location of all objects and a few regression heads to estimate the object size, rotation, and velocity. For map segmentation, different map categories may overlap ( _e.g_ ., crosswalk is a subset of drivable space). Therefore, we formulate this problem as multiple binary semantic segmentation, one for each class. We follow CVT [8] to train the segmentation head with the standard focal loss [54]. 

# _C. Fully-Convolutional Fusion_ 

With all sensory features converted to the shared BEV representation, we can easily fuse them together with an elementwise operator (such as concatenation). Though in the same space, LiDAR BEV features and camera BEV features can still be spatially misaligned to some extent due to the inaccurate depth in the view transformer. To this end, we apply a convolution-based BEV encoder (with a few residual 

# IV. EXPERIMENTS 

We evaluate BEVFusion for camera-LiDAR fusion on 3D object detection and BEV map segmentation, covering both geometric- and semantic-oriented tasks. Our framework can be easily extended to support other types of sensors (such as radars and event-based cameras) and other 3D perception tasks (such as 3D object tracking and motion forecasting). 

Tab. III: BEVFusion outperforms the state-of-the-art multi-sensor fusion methods by **13.6%** on BEV map segmentation on nuScenes (val) with consistent improvements across different categories. 

||Modality|Drivable|Ped. Cross.|Walkway|Stop Line|Carpark|Divider|Mean|
|---|---|---|---|---|---|---|---|---|
|OFT [38]|C|74.0|35.3|45.9|27.5|35.9|33.9|42.1|
|LSS [6]|C|75.4|38.8|46.3|30.3|39.1|36.5|44.4|
|CVT [8]|C|74.3|36.8|39.9|25.8|35.0|29.4|40.2|
|M<sup>2</sup>BEV [41]|C|77.2|–|–|–|–|40.5|–|
|**BEVFusion** (Ours)|C|**81.7**|**54.8**|**58.4**|**47.4**|**50.7**|**46.4**|**56.6**|
|PointPillars [10]|L|72.0|43.1|53.1|29.7|27.7|37.5|43.8|
|CenterPoint [17]|L|75.6|48.4|57.5|36.5|31.7|41.9|48.6|
|PointPainting [1]|C+L|75.9|48.5|57.1|36.9|34.5|41.9|49.1|
|MVP [4]|C+L|76.1|48.7|57.0|36.9|33.0|42.2|49.0|
|**BEVFusion** (Ours)|C+L|**85.5**|**60.5**|**67.6**|**52.0**|**57.0**|**53.7**|**62.7**|



Tab. IV: BEVFusion is robust under different lighting and weather conditions, significantly boosting the performance single-modality models under challenging rainy(+10.7) and nighttime(+12.8) scenes. 

|||Su|nny|Ra|iny|D|ay|N|ight|
|---|---|---|---|---|---|---|---|---|---|
||Modality|mAP|mIoU|mAP|mIoU|mAP|mIoU|mAP|mIoU|
|CenterPoint [17]|L|62.9|50.7|59.2|42.3|62.8|48.9|35.4|37.0|
|BEVFormer [43]|C|41.0|–|44.0|–|41.9|–|21.2|–|
|BEVFusion|C|–|59.0|–|50.5|–|57.4|–|30.8|
|MVP|C+L|65.9 (+3.0)|51.0 (+0.3)|66.3 (+7.1)|42.9 (+0.6)|66.3 (+3.5)|49.2 (+0.3)|38.4 (+3.0)|37.5 (+0.5)|
|BEVFusion|C+L|68.2 (+5.3)|65.6 (+6.6)|69.9 (+10.7)|55.9 (+5.4)|68.5 (+5.7)|63.1 (+5.7)|42.8 (+7.4)|43.6 (+12.8)|



**Model.** We use Swin-T [57] as our image backbone and VoxelNet [11] as our LiDAR backbone. We apply FPN [58] to fuse multi-scale camera features to produce a feature map of 1/8 input size. We downsample camera images to 256 _×_ 704 and voxelize the LiDAR point cloud with 0.075m (for detection) and 0.1m (for segmentation). As detection and segmentation tasks require BEV feature maps with different spatial ranges and sizes, we apply grid sampling with bilinear interpolation before each task-specific head to explicitly transform between different BEV feature maps. 

**Dataset.** We evaluate our method on nuScenes [59] and Waymo [60], which are large-scale datasets for 3D perception with _>_ 40k annotated scenes. Each sample in both datasets are equipped with both LiDAR and surrounding camera inputs. 

# _A. 3D Object Detection_ 

We first experiment on the geometric-centric 3D object detection benchmark, where BEVFusion achieves superior performance with lower computation cost and measured latency. We use the mean average precision (mAP) across 10 foreground classes and the nuScenes detection score (NDS) as our detection metrics. We also measure the single-inference #MACs and latency on an RTX3090 GPU for all opensource methods. We use a single model without any test-time augmentation for both val and test results. 

As in Table I, BEVFusion achieves state-of-the-art results on the nuScenes detection benchmark, with close-to-real-time ( **8.4 FPS** ) inference speed on a desktop GPU. Compared with TransFusion [49], BEVFusion offers 1.3% improvement in test split mAP and NDS, while significantly reduces the MACs by **1.9** _×_ and measured latency by **1.3** _×_ . It also compares favorably against representative point-level fusion 

methods PointPainting [1] and MVP [4] with **1.6** _×_ speedup, **1.5** _×_ MACs reduction and **3.8%** higher mAP on the test set. We argue that the efficiency gain of BEVFusion comes from the fact that we choose the BEV space as the shared fusion space, which fully utilizes all camera features instead of just a 5% sparse set. Consequently, BEVFusion can achieve the same performance with much smaller resolution for the camera inputs, resulting in significantly lower MACs. Combined with the efficient BEV pooling operator in Section III-B, BEVFusion transfers MACs reduction into measured speedup. 

BEVFusion also achieves state-of-the-art performance on the Waymo open dataset [60] (Table II). BEVFusion outperforms the previous state-of-the-art multi-modal detector, DeepFusion [3] with 60% of input frames. Furthermore, DeepFusion ensembles 25 models evaluated with test-time augmentation, while we deliver better performance by applying test-time augmentation to a single BEVFusion model. 

# _B. BEV Map Segmentation_ 

We further compare BEVFusion with state-of-the-art models on the semantic-centric BEV map segmentation task, where BEVFusion achieves an even larger performance boost. We report the Intersection-over-Union (IoU) on 6 background classes and the class-averaged mean IoU as our evaluation metric. As different classes may have overlappings ( _e.g_ . car-parking area is also drivable), we evaluate the binary segmentation performance for each class separately and select the highest IoU across different thresholds [8]. For each frame, we only perform the evaluation in the [-50m, 50m] _×_ [-50m, 50m] region around the ego car following [6], [8], [41], [43]. 

We report the BEV map segmentation results in Table III. In contrast to 3D object detection which is a _geometric_ -oriented task, map segmentation is _semantic_ -oriented. As a result, 



<!-- Start of picture text -->
LiDAR-onlyyy MVP BEVFusion<br>61.9<br>61.0<br>5.8%<br>better<br>57.9<br>56.1<br>4.2%<br>better<br>53.7<br>53.1<br><!-- End of picture text -->



<!-- Start of picture text -->
LiDAR-onlyyy MVP BEVFusion<br>61.9<br>61.0<br>5.8%<br>better<br>57.9<br>56.1<br>4.2%<br>better<br>53.7<br>53.1<br><!-- End of picture text -->



<!-- Start of picture text -->
LiDAR-onlyyy MVP BEVFusion<br>61.9<br>61.0<br>5.8%<br>better<br>57.9<br>56.1<br>4.2%<br>better<br>53.7<br>53.1<br><!-- End of picture text -->



<!-- Start of picture text -->
78.5<br>77.1<br>64.6<br>60.4<br>50.5<br>Improvementsprovementsrovements<br>0-20m 1.4<br>20-30m 4.2<br>>30m 7.3 43.23.22<br><!-- End of picture text -->



<!-- Start of picture text -->
78.5<br>77.1<br>64.6<br>60.4<br>50.5<br>Improvementsprovementsrovements<br>0-20m 1.4<br>20-30m 4.2<br>>30m 7.3 43.23.22<br><!-- End of picture text -->



<!-- Start of picture text -->
CenterPoint MVP BEVFusion<br>63.2 64.4<br>63.8<br>61.4<br>58.5<br>52.0 54.9<br>+12% MACs (G) @ 16 beam<br>CenterPoint 75.3<br>39.8<br>MVP 292.7<br>35.8 BEVFusion 186.1<br><!-- End of picture text -->



<!-- Start of picture text -->
CenterPoint MVP BEVFusion<br>63.2 64.4<br>63.8<br>61.4<br>58.5<br>52.0 54.9<br>+12% MACs (G) @ 16 beam<br>CenterPoint 75.3<br>39.8<br>MVP 292.7<br>35.8 BEVFusion 186.1<br><!-- End of picture text -->

Fig. 4: BEVFusion outperforms state-of-the-art single- and multi-modality detectors under different LiDAR sparsity, object sizes and object distances, especially under more challenging settings ( _i.e_ ., _sparser point clouds_ , _small/distant objects_ ). 

our camera-only BEVFusion model outperforms LiDAR-only baselines by **8-13%** . This observation is the exact opposite of results in Table I, where state-of-the-art camera-only 3D detectors got outperformed by LiDAR-only detectors by almost 20 mAP. Our camera-only model boosts the performance of existing monocular BEV map segmentation methods by at least **12%** . In the multi-modality setting, we further improve the performance of the monocular BEVFusion by **6** mIoU and achieved _>_ **13%** improvement over state-ofthe-art sensor fusion methods [1], [4]. This is because both baseline methods are _object_ -centric and _geometric_ -oriented. PointPainting [1] only decorates the _foreground_ LiDAR points and MVP only densifies _foreground_ 3D objects. Both approaches are not helpful for segmenting map components. Worse still, both methods assume that LiDAR should be the more effective modality in sensor fusion, which is not true according to our observations in Table III. 

# V. ANALYSIS 

We present in-depth analyses of BEVFusion over singlemodality models and state-of-the-art multi-modality models. 

**Weather and Lighting.** We first analyze the performance of BEVFusion under different weather and lighting conditions in Table IV. LiDAR-only models face significant challenges in detecting objects in rainy weather due to sensor noise, while BEVFusion leverages the robustness of camera sensors to achieve a **10.7** mAP improvement, which largely narrows the performance gap between sunny and rainy scenarios. Poor lighting conditions pose challenges for both detection and segmentation models. For detection, MVP’s improvement is relatively small compared to BEVFusion, which relies less on _accurate_ 2D instance segmentations to generate virtual points and therefore performs better in dark or overexposed scenes. For segmentation, while camera-only BEVFusion outperforms CenterPoint on the entire benchmark, its performance drops significantly at nighttime. However, multi-modal BEVFusion achieves a **12.8** mIoU improvement, even greater than its improvement in the daytime, demonstrating the importance of leveraging geometric clues when camera sensors fail. 

**Sizes and Distances.** We also analyze the performance of BEVFusion under different object sizes and distances. From Figure 4a, BEVFusion achieves consistent improvements over 

its LiDAR-only counterpart for both small and large objects, while MVP has only negligible improvements for objects larger than 4m. This is because larger objects are typically much denser, benefiting less from augmented multi-modal virtual points (MVPs). Additionally, BEVFusion yields greater improvements to the LiDAR-only detector for smaller objects (in Figure 4a) and more distant objects (in Figure 4b), both of which are inadequately captured by LiDAR and can therefore derive more benefit from the dense camera information. 

**Sparser LiDARs.** We finally demonstrate the performance of CenterPoint [17] (LiDAR-only), MVP [4] (multi-modal), and our BEVFusion under different LiDAR sparsities in Figure 4c. BEVFusion consistently outperforms MVP across all sparsity levels with a **1.6** _×_ reduction in #MACs and achieves a **12%** improvement in the 1-beam LiDAR scenario. MVP decorates the point cloud and directly applies CenterPoint on the painted and densified LiDAR input. As a result, it naturally requires the LiDAR-only detector (CenterPoint) to perform well, which is not valid under sparse LiDAR settings ( _i.e_ ., 35.8 NDS with 1-beam input in Figure 4c). In contrast, BEVFusion integrates multi-sensor information in the shared BEV space and does not rely solely on a robust LiDAR-only detector. 

# VI. CONCLUSION 

We present BEVFusion, an efficient and generic framework for multi-task multi-sensor 3D perception. BEVFusion unifies camera and LiDAR features in a shared BEV space that fully preserves geometric and semantic information. To achieve this, we accelerate the slow camera-to-BEV transformation by more than 40 times. BEVFusion rethinks the effectiveness of point-level fusion in multi-sensor perception systems and achieves superior performance on both nuScenes 3D detection and BEV map segmentation tasks with 1.5-1.9 _×_ less computation and 1.3-1.6 _×_ measured speedup over existing solutions. BEVFusion also outperforms all existing sensor fusion methods on Waymo open dataset. We hope that BEVFusion can serve as a simple but powerful baseline to inspire future research on multi-task multi-sensor fusion. 

# REFERENCES 

> [1] S. Vora, A. H. Lang, B. Helou, and O. Beijbom, “PointPainting: Sequential Fusion for 3D Object Detection,” in _CVPR_ , 2020. 

- [2] C. Wang, C. Ma, M. Zhu, and X. Yang, “PointAugmenting: CrossModal Augmentation for 3D Object Detection,” in _CVPR_ , 2021. 

- [3] Y. Li, A. W. Yu, T. Meng, B. Caine, J. Ngiam, D. Peng, J. Shen, B. Wu, Y. Lu, D. Zhou _et al._ , “DeepFusion: Lidar-Camera Deep Fusion for Multi-Modal 3D Object Detection,” in _CVPR_ , 2022. 

- [4] T. Yin, X. Zhou, and P. Krahenb¨ uhl,¨ “Multimodal Virtual Point 3D Detection,” in _NeurIPS_ , 2021. 

- [5] B. Pan, J. Sun, H. Y. T. Leung, A. Andonian, and B. Zhou, “Cross-View Semantic Segmentation for Sensing Surroundings,” _RA-L_ , 2020. 

- [6] J. Philion and S. Fidler, “Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D,” in _ECCV_ , 2020. 

- [7] Q. Li, Y. Wang, Y. Wang, and H. Zhao, “HDMapNet: An Online HD Map Construction and Evaluation Framework,” in _ICRA_ , 2022. 

- [8] B. Zhou and P. Krahenb¨ uhl,¨ “Cross-View Transformers for Real-Time Map-View Semantic Segmentation,” in _CVPR_ , 2022. 

- [9] Y. Zhou and O. Tuzel, “VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection,” in _CVPR_ , 2018. 

- [10] A. H. Lang, S. Vora, H. Caesar, L. Zhou, and J. Yang, “PointPillars: Fast Encoders for Object Detection from Point Clouds,” in _CVPR_ , 2019. 

- [11] Y. Yan, Y. Mao, and B. Li, “SECOND: Sparsely Embedded Convolutional Detection,” _Sensors_ , 2018. 

- [12] B. Zhu, Z. Jiang, X. Zhou, Z. Li, and G. Yu, “Class-Balanced Grouping and Sampling for Point Cloud 3D Object Detection,” _arXiv_ , 2019. 

- [13] Z. Yang, Y. Sun, S. Liu, and J. Jia, “3DSSD: Point-Based 3D Single Stage Object Detector,” _CVPR_ , 2020. 

- [14] Y. Zhou, P. Sun, Y. Zhang, D. Anguelov, J. Gao, T. Ouyang, J. Guo, J. Ngiam, and V. Vasudevan, “End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds,” _CoRL_ , 2019. 

- [15] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space,” in _NeurIPS_ , 2017. 

- [16] B. Graham, M. Engelcke, and L. van der Maaten, “3D Semantic Segmentation With Submanifold Sparse Convolutional Networks,” in _CVPR_ , 2018. 

- [17] T. Yin, X. Zhou, and P. Krahenb¨ uhl, “Center-Based 3D Object Detection¨ and Tracking,” in _CVPR_ , 2021. 

- [18] R. Ge, Z. Ding, Y. Hu, W. Shao, L. Huang, K. Li, and Q. Liu, “1<sup>st</sup> Place Solutions to the Real-time 3D Detection and the Most Efficient Model of the Waymo Open Dataset Challenge 2021,” in _CVPRW_ , 2021. 

- [19] Q. Chen, L. Sun, Z. Wang, K. Jia, and A. Yuille, “Object as Hotspots: An Anchor-Free 3D Object Detection Approach via Firing of Hotspots,” in _ECCV_ , 2020. 

- [20] C. R. Qi, Y. Zhou, M. Najibi, P. Sun, K. Vo, B. Deng, and D. Anguelov, “Offboard 3D Object Detection from Point Cloud Sequences,” in _CVPR_ , 2021. 

- [21] L. Fan, X. Xiong, F. Wang, N. Wang, and Z. Zhang, “RangeDet: In Defense of Range View for LiDAR-Based 3D Object Detection,” in _ICCV_ , 2021. 

- [22] Q. Chen, S. Vora, and O. Beijbom, “PolarStream: Streaming Lidar Object Detection and Segmentation with Polar Pillars,” in _NeurIPS_ , 2021. 

- [23] Y. Wang and J. M. Solomon, “Object DGCNN: 3D Object Detection using Dynamic Graphs,” in _NeurIPS_ , 2021. 

- [24] S. Shi, X. Wang, and H. Li, “PointRCNN: 3D Object Proposal Generation and Detection From Point Cloud,” in _CVPR_ , 2019. 

- [25] Y. Chen, S. Liu, X. Shen, and J. Jia, “Fast Point R-CNN,” in _ICCV_ , 2019. 

- [26] S. Shi, Z. Wang, J. Shi, X. Wang, and H. Li, “From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Partaggregation Network,” _TPAMI_ , 2020. 

- [27] S. Shi, C. Guo, L. Jiang, Z. Wang, J. Shi, X. Wang, and H. Li, “PVRCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection,” in _CVPR_ , 2020. 

- [28] S. Shi, L. Jiang, J. Deng, Z. Wang, C. Guo, J. Shi, X. Wang, and H. Li, “PV-RCNN++: Point-Voxel Feature Set Abstraction With Local Vector Representation for 3D Object Detection,” _arXiv_ , 2021. 

- [29] Z. Li, F. Wang, and N. Wang, “LiDAR R-CNN: An Efficient and Universal 3D Object Detector,” _CVPR_ , 2021. 

- [30] T. Wang, X. Zhu, J. Pang, and D. Lin, “FCOS3D: Fully Convolutional One-Stage Monocular 3D Object Detection,” in _ICCVW_ , 2021. 

- [31] Z. Tian, C. Shen, H. Chen, and T. He, “FCOS: Fully Convolutional One-Stage Object Detection,” in _ICCV_ , 2019. 

- [32] T. Wang, X. Zhu, J. Pang, and D. Lin, “Probabilistic and geometric depth: Detecting objects in perspective,” in _CoRL_ , 2021. 

- [33] H. Chen, P. Wang, F. Wang, W. Tian, L. Xiong, and H. Li, “EProPnP: Generalized End-to-End Probabilistic Perspective-n-Points for Monocular Object Pose Estimation,” in _CVPR_ , 2022. 

- [34] Y. Wang, V. Guizilini, T. Zhang, Y. Wang, H. Zhao, and J. M. Solomon, “DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries,” in _CoRL_ , 2021. 

- [35] Y. Liu, T. Wang, X. Zhang, and J. Sun, “PETR: Position Embedding Transformation for Multi-View 3D Object Detection,” _arXiv_ , 2022. 

- [36] X. Zhu, W. Su, L. Lu, B. Li, X. Wang, and J. Dai, “Deformable DETR: Deformable Transformers for End-to-End Object Detection,” in _ICLR_ , 2021. 

- [37] Y. Wang, X. Zhang, T. Yang, and J. Sun, “Anchor DETR: Query Design for Transformer-Based Detector,” in _AAAI_ , 2022. 

- [38] T. Roddick, A. Kendall, and R. Cipolla, “Orthographic Feature Transform for Monocular 3D Object Detection,” in _BMVC_ , 2019. 

- [39] T. Roddick and R. Cipolla, “Predicting Semantic Map Representations from Images using Pyramid Occupancy Networks,” in _CVPR_ , 2020. 

- [40] J. Huang, G. Huang, Z. Zhu, Y. Ye, and D. Du, “BEVDet: Highperformance Multi-camera 3D Object Detection in Bird-Eye-View,” _arXiv_ , 2021. 

- [41] E. Xie, Z. Yu, D. Zhou, J. Philion, A. Anandkumar, S. Fidler, P. Luo, and J. M. Alvarez, “M<sup>2</sup> BEV: Multi-Camera Joint 3D Detection and Segmentation with Unified Birds-Eye View Representation,” _arXiv_ , 2022. 

- [42] C. Reading, A. Harakeh, J. Chae, and S. L. Waslander, “Categorical depth distributionnetwork for monocular 3d object detection,” in _CVPR_ , 2021. 

- [43] Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu, Y. Qiao, and J. Dai, “BEVFormer: Learning Bird’s-Eye-View Representation from MultiCamera Images via Spatiotemporal Transformers,” _arXiv_ , 2022. 

- [44] X. Chen, H. Ma, J. Wan, B. Li, and T. Xia, “Multi-View 3D Object Detection Network for Autonomous Driving,” in _CVPR_ , 2017. 

- [45] C. R. Qi, W. Liu, C. Wu, H. Su, and L. J. Guibas, “Frustum PointNets for 3D Object Detection from RGB-D Data,” in _CVPR_ , 2018. 

- [46] Z. Wang and K. Jia, “Frustum ConvNet: Sliding Frustums to Aggregate Local Point-Wise Features for Amodal 3D Object Detection,” in _IROS_ , 2019. 

- [47] R. Nabati and H. Qi, “CenterFusion: Center-Based Radar and Camera Fusion for 3D Object Detection,” in _WACV_ , 2021. 

- [48] X. Chen, T. Zhang, Y. Wang, Y. Wang, and H. Zhao, “FUTR3D: A Unified Sensor Fusion Framework for 3D Detection,” _arXiv_ , 2022. 

- [49] X. Bai, Z. Hu, X. Zhu, Q. Huang, Y. Chen, H. Fu, and C.-L. Tai, “TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers,” in _CVPR_ , 2022. 

- [50] S. Xu, D. Zhou, J. Fang, J. Yin, B. Zhou, and L. Zhang, “FusionPainting: Multimodal Fusion with Adaptive Attention for 3D Object Detection,” in _ITSC_ , 2021. 

- [51] Z. Chen, Z. Li, S. Zhang, L. Fang, Q. Jiang, F. Zhao, B. Zhou, and H. Zhao, “AutoAlign: Pixel-Instance Feature Aggregation for MultiModal 3D Object Detection,” _arXiv_ , 2022. 

- [52] M. Liang, B. Yang, S. Wang, and R. Urtasun, “Deep Continuous Fusion for Multi-Sensor 3D Object Detection,” in _ECCV_ , 2018. 

- [53] D. Park, R. Ambrus, V. Guizilini, J. Li, and A. Gaidon, “Is Pseudo-Lidar needed for Monocular 3D Object detection?” in _ICCV_ , 2021. 

- [54] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar,´ “Focal Loss for Dense Object Detection,” in _ICCV_ , 2017. 

- [55] S. Chen, Z. Jie, X. Wei, and L. Ma, “MT-Net Submission to the Waymo 3D Detection Leaderboard,” _arXiv_ , 2022. 

- [56] X. Chen, S. Shi, B. Zhu, K. C. Cheung, H. Xu, and H. Li, “MPPNet: Multi-Frame Feature Intertwining with Proxy Points for 3D Temporal Object Detection,” in _ECCV_ , 2022. 

- [57] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows,” in _ICCV_ , 2021. 

- [58] T.-Y. Lin, P. Dollar,´ R. Girshick, K. He, B. Hariharan, and S. Belongie, “Feature Pyramid Networks for Object Detection,” in _CVPR_ , 2017. 

- [59] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuScenes: A Multimodal Dataset for Autonomous Driving,” in _CVPR_ , 2020. 

- [60] P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine, V. Vasudevan, W. Han, J. Ngiam, H. Zhao, A. Timofeev, S. Ettinger, M. Krivokon, A. Gao, A. Joshi, 

Y. Zhang, J. Shlens, Z. Chen, and D. Anguelov, “Scalability in Perception for Autonomous Driving: Waymo Open Dataset,” in _CVPR_ . 

