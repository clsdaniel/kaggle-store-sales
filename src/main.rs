use burn::{backend::Autodiff, data::{dataloader::{DataLoaderBuilder, batcher::Batcher}, dataset::InMemDataset}, optim::AdamConfig, prelude::*, record::CompactRecorder, train::{Learner, SupervisedTraining, metric::LossMetric}};
use muonts::{data::batchitem::BatchItem, models::tft::model::TemporalFusionTransformerModelConfig};
use polars::prelude::*;
use burn_flex::{Flex, FlexDevice};
use rand::{Rng, seq::IndexedRandom};

type MyBackend = Autodiff<Flex>;

pub struct TimeseriesBatcher<B: Backend> {
    context_length: usize,
    prediction_length: usize,
    _marker: std::marker::PhantomData<B>
}

impl<B: Backend> TimeseriesBatcher<B> {
    pub fn new(context_length: usize, prediction_length: usize) -> TimeseriesBatcher<B> {
        TimeseriesBatcher::<B> { context_length, prediction_length, _marker: std::marker::PhantomData }
    }
}

impl<B: Backend> Batcher<B, Vec<f64>, BatchItem<B>> for TimeseriesBatcher<B> {
    fn batch(&self, items: Vec<Vec<f64>>, device: &B::Device) -> BatchItem<B> {
        let batch_size = items.len();

        let (past_vec, future_vec): (Vec<Vec<f64>>, Vec<Vec<f64>>) = items.iter()
        .map(|ts| {
            let n = ts.len();
            let start = n - (self.context_length + self.prediction_length);
            let past = ts[start..start+self.context_length].to_vec();
            let future = ts[start+self.context_length..].to_vec();

            (past, future)
        }).unzip();

        let past_target = Tensor::from_data(TensorData::new(past_vec.into_iter().flatten().collect(), [batch_size, self.context_length]), device);
        let past_observed_values = Tensor::ones([batch_size, self.context_length], device);

        let future_target = Tensor::from_data(TensorData::new(future_vec.into_iter().flatten().collect(), [batch_size, self.prediction_length]), device);        
        let future_observed_values = Tensor::ones([batch_size, self.prediction_length], device);

        let feat_static_real = Tensor::ones([batch_size, 1], device);
        let feat_static_cat = Tensor::ones([batch_size, 1], device);
        let feat_dynamic_real = Tensor::ones([batch_size, self.context_length + self.prediction_length, 1], device);

        BatchItem { 
            past_target, 
            past_observed_values, 
            future_target, 
            future_observed_values, 
            feat_static_real: Some(feat_static_real), 
            feat_static_cat: Some(feat_static_cat), 
            feat_dynamic_real: Some(feat_dynamic_real), 
            feat_dynamic_cat: None, 
            past_feat_dynamic_real: None, past_feat_dynamic_cat: None 
        }
    }
}

fn main() -> anyhow::Result<()> {
    let context_length : usize = 56;
    let prediction_length : usize = 28;
    let batch_size = 32;
    let num_epochs : usize = 10;
    let learning_rate = 1e-3;

    let artifact_dir = "./target/kaggle_tft_artifacts";
    
    // Create artifact folder and clean old ones first
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();

    let df = LazyCsvReader::new(PlRefPath::new("data/train.csv"))
        .finish()
        .unwrap();

    let data = df.group_by([col("store_nbr"), col("family")])
        .agg([col("date"), col("sales")])
        .collect()
        .unwrap();

    let column = data.column("sales")?.list()?;
    
    let data_vec: Vec<Vec<f64>> = column.series_iter()
        .filter_map(|opt_series| {
            let s = opt_series?;
            let ca = s.f64().ok()?;
            Some(ca.iter().flatten().collect())
        })
        .collect();

    let mut rng = rand::rng();
    let data_valid: Vec<Vec<f64>> = data_vec.sample(&mut rng, data_vec.len() / 10).cloned().collect();

    let data_pred: Vec<Vec<f64>> = data_vec.sample(&mut rng, data_vec.len() / 10).cloned().collect();

    let dataset = InMemDataset::new(data_vec);
    let batcher_train = TimeseriesBatcher::<MyBackend>::new(context_length, prediction_length);
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(4)
        .build(dataset);

    let dataset_valid = InMemDataset::new(data_valid);
    let batcher_valid = TimeseriesBatcher::<Flex>::new(context_length, prediction_length);
    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(4)
        .build(dataset_valid);

    let device = &FlexDevice;

    let model = TemporalFusionTransformerModelConfig::new(context_length, prediction_length)
        .with_c_feat_static_cat(vec![])
        .with_d_feat_static_real(vec![1])
        .with_d_feat_dynamic_real(vec![1])
        .init::<MyBackend>(device);
    
    let optim = AdamConfig::new().init();

    let trainer = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
        .metrics((LossMetric::new(),))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(num_epochs)
        .summary();

    let result = trainer.launch(Learner::new(
        model,
        optim,
        learning_rate,
    ));

    let inference_model = result.model;
    let pred_batcher = TimeseriesBatcher::<Flex>::new(context_length, prediction_length);
    let mut writer = csv::Writer::from_path("prediction.csv")?;
    writer.write_record(&["id", "idx", "kind", "value"])?;

    data_pred.into_iter()
        .enumerate()
        .for_each(|(idx, ts)| {
            let start_n = ts.len() - prediction_length;

            ts.iter().enumerate().for_each(|(n, v)|{
                writer.write_record(&[idx.to_string(), n.to_string(), "historical".to_string(), v.to_string()]);
            });

            let batch = pred_batcher.batch(vec![ts], device);
            let preds = inference_model.forward(batch.past_target, 
                batch.past_observed_values, 
                batch.feat_static_real, 
                batch.feat_static_cat, 
                batch.feat_dynamic_real, 
                batch.feat_dynamic_cat, 
                batch.past_feat_dynamic_real, 
                batch.past_feat_dynamic_cat);
            
            let preds_flat: Vec<f32> = preds.into_data().iter::<f32>().collect();
            
            let p10 = &preds_flat[0..prediction_length];
            let p50 = &preds_flat[prediction_length..2 * prediction_length];
            let p90 = &preds_flat[2 * prediction_length..3 * prediction_length];

            p10.iter().enumerate().for_each(|(n, v)| {
                writer.write_record(&[idx.to_string(), (start_n+n).to_string(), "p10".to_string(), v.to_string()]);
            });
            p50.iter().enumerate().for_each(|(n, v)| {
                writer.write_record(&[idx.to_string(), (start_n+n).to_string(), "p50".to_string(), v.to_string()]);
            });
            p90.iter().enumerate().for_each(|(n, v)| {
                writer.write_record(&[idx.to_string(), (start_n+n).to_string(), "p90".to_string(), v.to_string()]);
            });
        });

    writer.flush();

    Ok(())
}

