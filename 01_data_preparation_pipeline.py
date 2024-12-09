import kfp
import kfp.dsl as dsl

from data_components import (
    download_ml25m_data,
    unzip_data,
    csv_to_parquet,
    split_dataset,
    put_to_minio,
    qa_data
)

@dsl.pipeline(
  name='Data prep pipeline',
  description='A pipeline that retrieves data from movielens and ingests it into paraquet files on minio'
)
def dataprep_pipeline(minio_bucket:str='datasets', random_init:int=42):
    download_dataset = download_ml25m_data()
    unzip_folder = unzip_data(input_path=download_dataset.outputs['output_path_one'])
    ratings_parquet_op = csv_to_parquet(inputFile=unzip_folder.outputs['ratings_output_path'])
    movies_parquet_op = csv_to_parquet(inputFile=unzip_folder.outputs['movies_output_path'])
    split_op = split_dataset(input_parquet=ratings_parquet_op.output,random_state=random_init)
    u1 = put_to_minio(inputFile=movies_parquet_op.output, upload_file_name='movies.parquet.gzip', bucket=minio_bucket)
    u2 = put_to_minio(inputFile=split_op.output, bucket=minio_bucket)
    qa_op = qa_data(bucket=minio_bucket).after(u2)

    download_dataset.set_caching_options(False)
    unzip_folder.set_caching_options(False)
    ratings_parquet_op.set_caching_options(False)
    movies_parquet_op.set_caching_options(False)
    split_op.set_caching_options(False)
    u1.set_caching_options(False)
    u2.set_caching_options(False)
    qa_op.set_caching_options(False)
    

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=dataprep_pipeline,
        package_path='compiled_pipelines/dataPrep_pipeline.yaml')