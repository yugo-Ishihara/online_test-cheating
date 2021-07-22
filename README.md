# online_test-cheating

オンライン試験における不正行為検出システム
　オンライン試験における替え玉受験やカンニング等の不正行為を検出する
使用したモデル
　action-recognition-0001-encoder
　　→受験者の動作から特徴量を抽出
　action-recognition-0001-decoder
　　→抽出された特徴量から動作を推定する
　face-detection-retail-0004
　　→顔検出
　age-gender-recognition-retail-0013
　　→年齢性別推定
不正行為を識別する
　PC内蔵のカメラを用いて受験者の動作推定を行う
　動作が不正行為と見なされた場合、画面上に不正行為を示すテキストが表示される
本人確認
　PC内蔵のカメラを用いて受験者の年齢・性別推定を行う
　顔を識別し、顔の上に年齢と性別情報が表示される
　　→替え玉受験の抑止力
