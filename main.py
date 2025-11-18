import argparse
import os
import sys
import itertools

# اضافه کردن مسیر مدل‌ها به Python path
sys.path.append(os.path.abspath("SuperGluePretrainedNetwork-master"))

# وارد کردن ماژول‌ها
from GroundTruthGenerator import GroundTruthGenerator


def main():
    parser = argparse.ArgumentParser(description='Run grid search for SuperPoint + SuperGlue configurations.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Root folder containing "rgb" and "thermal" folders.')
    parser.add_argument('--output_dir', type=str, required=True, help='Folder for saving results.')
    parser.add_argument('--backend_dir', type=str, required=True, help='Directory for backend cache.')
    parser.add_argument('--sp_model_path', type=str, required=True, help='Path to the SuperPoint model weights.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for stability in score calculation.')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- محدوده‌ی پارامترهای جست‌وجو ---
    nms_radius_values = [3, 4, 5]
    keypoint_threshold_values = [0.005, 0.01, 0.02]
    max_keypoints_values = [2048]
    sinkhorn_iterations_values = [20, 50]
    match_threshold_values = [0.1, 0.2]

    alpha = args.alpha  # وزن ثبات

    # --- گرفتن شناسه‌ی تایل‌ها از پوشه‌ی rgb ---
    rgb_dir = os.path.join(args.dataset_dir, "rgb")
    tile_ids = [os.path.splitext(f)[0].replace("satellite_", "")
                for f in os.listdir(rgb_dir) if f.endswith(".png")]
    # print(tile_ids)

    results = []
    best_score = float('-inf')
    best_config = None

    # --- شروع گرید سرچ ---
    for nms_radius, keypoint_threshold, max_kp, sink_it, match_thr in itertools.product(
            nms_radius_values, keypoint_threshold_values, max_keypoints_values,
            sinkhorn_iterations_values, match_threshold_values):

        sp_config = {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_kp
        }

        sg_config = {
            'weights': 'outdoor',
            'sinkhorn_iterations': sink_it,
            'match_threshold': match_thr
        }

        try:
            gt_gen = GroundTruthGenerator(
                data_dir=args.dataset_dir,
                backend_dir=args.backend_dir,
                pair_ids=tile_ids,
                debug=True,
                sp_model_path=args.sp_model_path,
                sp_config=sp_config,
                sg_config=sg_config
            )

            
            matches = gt_gen.run()  
            
            print(f"[DEBUG]: matches -> {matches}")
            

            # میانگین و انحراف معیار بین تایل‌ها
            mean_matches = sum(matches) / len(matches)
            std_matches = (sum((x - mean_matches) ** 2 for x in matches) / len(matches)) ** 0.5

            score = mean_matches - alpha * std_matches

            results.append({
                'sp_config': sp_config,
                'sg_config': sg_config,
                'matches_per_tile': matches,
                'mean': mean_matches,
                'std': std_matches,
                'score': score
            })

            # به‌روزرسانی بهترین کانفیگ
            if score > best_score:
                best_score = score
                best_config = (sp_config, sg_config)

            print(f"[INFO] Config tested. Mean={mean_matches:.2f}, Std={std_matches:.2f}, Score={score:.2f}")


        except Exception as e:
            print(f"⚠️ Error during config test: {e}")

    # --- ذخیره‌ی نتایج ---
    all_results_path = os.path.join(args.output_dir, "all_results.txt")
    with open(all_results_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(f"SP: {res['sp_config']}, SG: {res['sg_config']}\n")
            f.write(f"Matches per tile: {res['matches_per_tile']}\n")
            f.write(f"Mean: {res['mean']}, Std: {res['std']}, Score: {res['score']}\n")
            f.write("-" * 60 + "\n")

    # --- ذخیره‌ی بهترین نتیجه ---
    best_result_path = os.path.join(args.output_dir, "best_result.txt")
    with open(best_result_path, "w", encoding="utf-8") as f:
        f.write(f"Best SP config: {best_config[0]}\n")
        f.write(f"Best SG config: {best_config[1]}\n")
        f.write(f"Best score: {best_score}\n")

    print("✅ Finished! Best score:", best_score)
    print(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
