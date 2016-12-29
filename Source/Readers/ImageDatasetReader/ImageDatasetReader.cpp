//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"

#include "ImageDatasetReader.h"

#include "DataReader.h"
#include "dataset_io.hpp"
#include "dataset_events_sink.hpp"
#include "FramePacker.h"
#include "ImageDatasetConfigHelper.h"
#include "Transformer.h"

#include <assert.h>
#include <memory>
#include <numeric>
#include <vector>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

// Implementation of interface required by data loader.
class ImageDatasetExample : public IExample<float>
{
public:
    // We expect image like blobs/streams from dataset.
    static const int c_blob_dims = 3;

    ImageDatasetExample(const vector<string>& blobNames)
    {
        // blobNames are list of blob names we are interested in.
        if (blobNames.size() == 0)
        {
            RuntimeError("Empty blob names vector provided.");
        }
        // Save blob names and resize other vectors appropriately.
        m_blobNames = blobNames;
        m_blobs.resize(blobNames.size());
        m_blob_shapes.resize(blobNames.size());
    }

    virtual void ReshapeBlob(int index, int channels, int height, int width) override
    {
        // This method is called by dataset loader before asking for memory to copy blob to.
        static_assert(c_blob_dims == 3, "Invalid blob dims.");
        m_blob_shapes[index][0] = channels;
        m_blob_shapes[index][1] = height;
        m_blob_shapes[index][2] = width;
        m_blobs[index].resize(channels * height * width);
    }

    virtual float* GetBlobMemory(int index) override
    {
        // Called by dataset loader. Blob memory will be copied here.
        return m_blobs[index].data();
    }

    const array<int, c_blob_dims>* GetBlobShape(const string& blobName)
    {
        // Return shape for the blob with the given name.
        int index = BlobIndexFromName(blobName);
        return &m_blob_shapes[index];
    }

    // Swaps contents of the blob with the given name with the given vector.
    const void SwapBlobData(const string& blobName, vector<float>& outer)
    {
        int index = BlobIndexFromName(blobName);
        m_blobs[index].swap(outer);
    }

private:
    // Helper method that returns blob index based on blob name.
    int BlobIndexFromName(const string& blobName)
    {
        int index = -1;
        for (size_t ib = 0; ib < m_blobNames.size(); ib++)
        {
            if (blobName == m_blobNames[ib])
            {
                index = static_cast<int>(ib);
                break;
            }
        }
        if (index == -1)
        {
            RuntimeError("Blob with name %s not found.", blobName.c_str());
        }
        return index;
    }

private:
    // List of the all blob names we are interested in.
    vector<string> m_blobNames;
    // Memory for all blobs.
    vector<vector<float> > m_blobs;
    // Shapes fro all the blobs.
    vector<array<int, c_blob_dims> > m_blob_shapes;
};

// We extend dense sequence base and add memory we own to it.
struct DenseSequenceDataIds : public DenseSequenceData
{
public:
    virtual const void* GetDataBuffer() override
    {
        return m_ownedData.data();
    }
    vector<float> m_ownedData;
};
typedef shared_ptr<DenseSequenceDataIds> DenseSequenceDataIdsPtr;

// We extend sparse sequence base and add nonzero indices and values memory.
struct SparseSequenceDataIds : public SparseSequenceData
{
public:
    virtual const void* GetDataBuffer() override
    {
        return m_valuesMemory.data();
    }
    vector<IndexType> m_indicesMemory;
    vector<float> m_valuesMemory;
};
typedef shared_ptr<SparseSequenceDataIds> SparseSequenceDataIdsPtr;

///////////////////////////////////////////////////////////////////////////////
//                      DATASET EVENTS IMPLEMENATION
///////////////////////////////////////////////////////////////////////////////
class DatasetEventsSinkImpl : public DatasetEventsSink
{
public:
  virtual void DataReadThreadsCount(int /*count*/) override
  {
    // TODO: add perf markers.
  }

  virtual void DataReadStart(int /*data_read_thread_id*/) override
  {
    // TODO: add perf markers.
  }

  virtual void DataReadEnd(int /*data_read_thread_id*/, size_t /*bytes_read*/) override
  {
    // TODO: add perf markers.
  }

  virtual void ImageProcessingThreadsCount(int /*count*/) override
  {
    // TODO: add perf markers.
  }

  virtual void ImageProcessingStart(int /*im_proc_thread_id*/) override
  {
    // TODO: add perf markers.
  }

  virtual void ImageProcessingEnd(int /*im_proc_thread_id*/) override
  {
    // TODO: add perf markers.
  }
};

///////////////////////////////////////////////////////////////////////////////
//                      DATA SOURCE IMPLEMENATION
///////////////////////////////////////////////////////////////////////////////

// Object that connects packer (which creates final data batch) and dataset
// loader. Packer requires SequenceEnumerator from which it pull data so we extend that
// interface to be able to communicate with packer.
class DataSource : public SequenceEnumerator
{
public:
    DataSource(const ConfigParameters& config)
        : m_epochSize(0), m_currEpochSampleCount(0)
    {
        m_currEpochSampleCount = 0;

        unique_ptr<DatasetEventsSinkImpl> events_sink = make_unique<DatasetEventsSinkImpl>();

        // mIoU workaround
        m_epochOverride = false;
        if (config.Exists(L"epochOverride"))
        {
            m_epochOverride = true;
        }

        // Collect runtime params.
        vector<OverridableParam> runtimeParameters;
        m_workerRank = 0;
        m_numberOfWorkers = 1;
        if (ImageDatasetConfigHelper::HasWorkerRank(config))
        {
            m_workerRank = ImageDatasetConfigHelper::GetWorkerRank(config);
            if (!m_epochOverride)
            {
                runtimeParameters.push_back({ OverridableParamID::loader_index, { to_string(m_workerRank) } });
            }
            // else: mIoU workaround: We want all readers to go through entire set, leave default value (this reader will think he is the only one
            // and go through entire set). This enables correct mIoU reporting.
        }
        if (ImageDatasetConfigHelper::HasWorkersCount(config))
        {
            m_numberOfWorkers = ImageDatasetConfigHelper::GetWorkersCount(config);
            if (!m_epochOverride)
            {
                runtimeParameters.push_back({ OverridableParamID::loaders_count, { to_string(m_numberOfWorkers) } });
            }
            // else: mIoU workaround: We want all readers to go through entire set, leave default value (this reader will think he is the only one
            // and go through entire set). This enables correct mIoU reporting.
        }
        if (ImageDatasetConfigHelper::HasDatasetDir(config))
        {
            runtimeParameters.push_back({ OverridableParamID::source_path, { ImageDatasetConfigHelper::GetDatasetDir(config) } });
        }
        if (ImageDatasetConfigHelper::HasIdsFiles(config))
        {
            // We have list of ids files specified, take it.
            string idsFiles = ImageDatasetConfigHelper::GetIdsFiles(config);
            // List is expected to be '|' separated, perform parsing.
            const char c_idsFileSeparator = '|';
            vector<string> idsFilesVec;
            stringstream idsFilesStream(idsFiles);
            string idsFile;
            while (getline(idsFilesStream, idsFile, c_idsFileSeparator))
            {
                idsFilesVec.emplace_back(idsFile);
            }
            runtimeParameters.push_back({ OverridableParamID::source_name, idsFilesVec });
        }

        // Kick off loading the dataset.
        m_dsLoader = CreateLoader<float>(ImageDatasetConfigHelper::GetLoadConfigPath(config), &runtimeParameters, move(events_sink));

        // Take names of the blobs inside dataset.
        int blobsCount = m_dsLoader->GetBlobsCount();
        vector<string> blobNames;
        for (int ib = 0; ib < blobsCount; ib++)
        {
            blobNames.emplace_back(m_dsLoader->GetBlobName(ib));
        }

        // Take one example to be used for tensor shape retrieval.
        m_example = make_unique<ImageDatasetExample>(blobNames);
        m_dsLoader->GetExample(m_example.get());

        // Create input and output streams. Stream is identical to required blob outputs from this reader.
        m_streamDescriptors = ImageDatasetConfigHelper::GetStreamDescriptors(config);
        int max_dimension = 0;
        int stream_id_in = 0;
        int stream_id_out = 0;
        for (size_t ib = 0; ib < m_streamDescriptors.size(); ib++)
        {
            const StreamDescriptor& streamDescriptor = m_streamDescriptors[ib];

            // Ensure we have blob with given name in dataset.
            if (find(blobNames.begin(), blobNames.end(), streamDescriptor.datasetName) == blobNames.end())
            {
                RuntimeError("Blob with name %s not found in image dataset.", streamDescriptor.datasetName.c_str());
            }

            // Take blob shape to be able to provide tensor shape.
            array<int, 3> shape = *m_example->GetBlobShape(streamDescriptor.datasetName);
            // Shape provided by image dataset has last dimension last, here we need last dimension first.
            reverse(shape.begin(), shape.end());

            // Create new input stream description and fill in the required fields.
            StreamDescriptionPtr inputStreamDescription = make_shared<StreamDescription>();
            m_inputStreams.push_back(inputStreamDescription);
            inputStreamDescription->m_id = stream_id_in++;
            inputStreamDescription->m_name = msra::strfun::utf16(streamDescriptor.name);
            inputStreamDescription->m_elementType = ElementType::tfloat;
            inputStreamDescription->m_storageType = streamDescriptor.datasetStorageType;
            if (streamDescriptor.datasetStorageType == StorageType::sparse_csc)
            {
                // In case of sparse data we expect one value in last dimension.
                if (shape[2] != 1)
                {
                    RuntimeError("Invalid image dataset shape for sparse data.");
                }
                // Final layout of the sample is dense, its last dimension must be declared in config.
                inputStreamDescription->m_sampleLayout = make_shared<TensorShape>(shape[0], shape[1], streamDescriptor.dimension);
                if (streamDescriptor.dimension > max_dimension)
                {
                    max_dimension = streamDescriptor.dimension;
                }

                // Check if input stream produces ignore label stream as well.
                if (streamDescriptor.ignoreStream != nullptr)
                {
                    StreamDescriptionPtr inputStreamDescriptionIgnore = make_shared<StreamDescription>();
                    m_inputStreams.push_back(inputStreamDescriptionIgnore);
                    inputStreamDescriptionIgnore->m_id = stream_id_in++;
                    inputStreamDescriptionIgnore->m_name = msra::strfun::utf16(streamDescriptor.ignoreStream->ignoreStreamName);
                    inputStreamDescriptionIgnore->m_elementType = ElementType::tfloat;
                    // Ignore stream is always dense.
                    inputStreamDescriptionIgnore->m_storageType = StorageType::dense;
                    inputStreamDescriptionIgnore->m_sampleLayout = make_shared<TensorShape>(shape[0], shape[1], 1);
                }
            }
            else
            {
                // Shape is equal to one pulled from the dataset.
                inputStreamDescription->m_sampleLayout = make_shared<TensorShape>(shape[0], shape[1], shape[2]);
            }

            // Create new output stream description and fill in the required fields. Same as previous one just storage
            // type must be dense.
            StreamDescriptionPtr outputStreamDescription = make_shared<StreamDescription>();
            m_outputStreams.push_back(outputStreamDescription);
            outputStreamDescription->m_id = stream_id_out++;
            outputStreamDescription->m_name = msra::strfun::utf16(streamDescriptor.name);
            outputStreamDescription->m_elementType = ElementType::tfloat;
            outputStreamDescription->m_storageType = StorageType::dense;
            if (streamDescriptor.datasetStorageType == StorageType::sparse_csc)
            {
                outputStreamDescription->m_sampleLayout = make_shared<TensorShape>(shape[0], shape[1], streamDescriptor.dimension);
                if (streamDescriptor.ignoreStream != nullptr)
                {
                    StreamDescriptionPtr outputStreamDescriptionIgnore = make_shared<StreamDescription>();
                    m_outputStreams.push_back(outputStreamDescriptionIgnore);
                    outputStreamDescriptionIgnore->m_id = stream_id_out++;
                    outputStreamDescriptionIgnore->m_name = msra::strfun::utf16(streamDescriptor.ignoreStream->ignoreStreamName);
                    outputStreamDescriptionIgnore->m_elementType = ElementType::tfloat;
                    // Ignore stream is always dense.
                    outputStreamDescriptionIgnore->m_storageType = StorageType::dense;
                    outputStreamDescriptionIgnore->m_sampleLayout = make_shared<TensorShape>(shape[0], shape[1], 1);
                }
            }
            else
            {
                outputStreamDescription->m_sampleLayout = make_shared<TensorShape>(shape[0], shape[1], shape[2]);
            }

        }
    }

    virtual vector<StreamDescriptionPtr> GetStreamDescriptions() const override
    {
        // Delegate call to accessor.
        return GetInputStreamDescriptions();
    }

    vector<StreamDescriptionPtr> GetInputStreamDescriptions() const
    {
        // Return input stream description.
        return m_inputStreams;
    }

    vector<StreamDescriptionPtr> GetOutputStreamDescriptions() const
    {
        // Return output stream description.
        return m_outputStreams;
    }

    virtual void StartEpoch(const EpochConfiguration& config) override
    {
        // Check that we read all examples from the previous epoch.
        if (m_epochSize != m_currEpochSampleCount)
        {
            RuntimeError("New epoch started without reading all samples from previous epoch (%zd != %zd).",
                m_epochSize, m_currEpochSampleCount);
        }
        // Check workers info.
        if (m_workerRank != config.m_workerRank)
        {
            RuntimeError("Rank changed in image dataset reader.");
        }
        if (m_numberOfWorkers != config.m_numberOfWorkers)
        {
            RuntimeError("Number of workers changed in image dataset reader.");
        }
        // Save current minibatch size.
        m_minibatchSize = config.m_minibatchSizeInSamples;
        // Check that minibatch size is divisible by number of workers.
        if (m_minibatchSize % m_numberOfWorkers != 0)
        {
            RuntimeError("Minibatch size (%zd) not divisible by number of workers (%zd).", m_minibatchSize, m_numberOfWorkers);
        }
        if (UseAllExamplesFromDatasetForEpoch(config))
        {
            if (m_epochOverride)
            {
                // mIoU workaround: Here we want all readers to go through entire set.
                m_epochSize = m_numberOfWorkers * static_cast<size_t>(m_dsLoader->GetExamplesCount());
            }
            else
            {
                // We take all examples from dataset for one epoch.
                m_epochSize = static_cast<size_t>(m_dsLoader->GetExamplesCount());
            }
        }
        else
        {
            // We take given number of examples for one epoch.
            m_epochSize = config.m_totalEpochSizeInSamples;
        }
        // Determine our portion of epoch taking into account number of distributed readers and minibatch size.
        // First take total number of full minibatches per reader.
        m_appendLastMinibatch = false;
        size_t thisReaderEpochSize = ((m_epochSize / m_minibatchSize) * m_minibatchSize) / m_numberOfWorkers;
        if (m_epochSize % m_minibatchSize != 0)
        {
            // We have rest of samples smaller than one minibatch, distribute them evenly.
            size_t partOfMiniBatch = ((m_epochSize % m_minibatchSize) / m_numberOfWorkers);
            bool allWorkersActiveInLastMinibatch = true;
            if (partOfMiniBatch == 0)
            {
                // We will not have enough data for the last minibatch for all workers, append this last samples to next
                // to last minibatch.
                allWorkersActiveInLastMinibatch = false;
            }
            thisReaderEpochSize += partOfMiniBatch;
            if ((m_epochSize % m_minibatchSize) % m_numberOfWorkers != 0)
            {
                // We still have couple of samples left (< m_numberOfWorkers), add them to first readers (by rank).
                if (m_workerRank < (m_epochSize % m_minibatchSize) % m_numberOfWorkers)
                {
                    thisReaderEpochSize++;
                    if (!allWorkersActiveInLastMinibatch)
                    {
                        // We do not have enough data to keep all workers busy at the last minibatch, append it to next to last.
                        m_appendLastMinibatch = true;
                    }
                }
            }
        }
        m_epochSize = thisReaderEpochSize;
        // We start epoch with 0 read samples.
        m_currEpochSampleCount = 0;
    }

    // Sets current configuration.
    virtual void SetConfiguration(const ReaderConfiguration& config) override
    {
        // Just check that nothing changed since StartEpoch was called.
        if (config.m_numberOfWorkers != m_numberOfWorkers)
        {
            RuntimeError("Number of workers changed since StartEpoch %zd != %zd.", config.m_numberOfWorkers, m_numberOfWorkers);
        }
        if (config.m_workerRank != m_workerRank)
        {
            RuntimeError("Workers rank changed since StartEpoch %zd != %zd.", config.m_workerRank, m_workerRank);
        }
        if (config.m_minibatchSizeInSamples != m_minibatchSize)
        {
            RuntimeError("Minibatch size changed since StartEpoch %zd != %zd.", config.m_minibatchSizeInSamples, m_minibatchSize);
        }
    }

    // Set current sample position
    virtual void SetCurrentSamplePosition(size_t /*currentSamplePosition*/) override
    {
        // Not needed currently.
        // TODO(VSO/OS/ANALOG_SL/#9698049): Investigate where/how this is used and implement missing logic.
    }

    // Returns current position in the global timeline. The returned value is in samples.
    virtual size_t GetCurrentSamplePosition() override
    {
        // Not needed currently.
        // TODO(VSO/OS/ANALOG_SL/#9698049): Investigate where/how this is used and implement missing logic.
        return 0;
    }

    virtual Sequences GetNextSequences(size_t totalSampleCount) override
    {
        // This method needs to return final (output) data in a form of set of sequences.
        Sequences sequences;

        // We expect that we are asked for number of samples equal to minibatch size.
        if (totalSampleCount != m_minibatchSize)
        {
            RuntimeError("Mismatch between minibatch size (%zd) and demanded sample count (%zd)", m_minibatchSize, totalSampleCount);
        }

        // Calculate sample count considering number of workers.
        size_t sampleCount = totalSampleCount / m_numberOfWorkers;
        if (sampleCount == 0)
        {
            RuntimeError("Greater number of workers than samples in minibatch.");
        }
        // So far we have sampleCount as if we deal with full minibatch, now we need to check corner case (end of epoch
        // where we may not have full minibatch).
        size_t remainingEpochSamples = m_epochSize - m_currEpochSampleCount;
        if (m_appendLastMinibatch && (remainingEpochSamples <= 2 * sampleCount))
        {
            // If we are appending we need to have one more sample.
            if (remainingEpochSamples != sampleCount + 1)
            {
                RuntimeError("Appending more than one sample (last minibatch size=%zd) to the last minibatch (minibatch size=%zd).",
                    remainingEpochSamples, sampleCount);
            }
            // We are at the next to last minibatch and we need to process all remaining samples (merge with last).
            sampleCount = remainingEpochSamples;
            sequences.m_endOfEpoch = true;
        }
        else if (!m_appendLastMinibatch && (remainingEpochSamples <= sampleCount))
        {
            // We are at the last minibatch, process what we have till end.
            sampleCount = remainingEpochSamples;
            sequences.m_endOfEpoch = true;
        }

        sequences.m_data.resize(m_inputStreams.size());

        // For each sequence we provide several streams.
        for (int istr = 0; istr < sequences.m_data.size(); istr++)
        {
            sequences.m_data[istr].resize(sampleCount);
        }

        // Now fill in the sequence data one by one.
        for (size_t ismpl = 0; ismpl < sampleCount; ismpl++)
        {
            // Go over the streams of the sequence.
            size_t istr = 0;
            size_t iStreamDesc = 0;
            while (istr < m_inputStreams.size())
            {
                // Take stream descriptor for the current sequence.
                const StreamDescriptor& streamDescriptor = m_streamDescriptors[iStreamDesc];
                // Increment stream descriptor index for the next cycle.
                iStreamDesc++;

                // We need to store different sequence object based on storage type.
                if (m_inputStreams[istr]->m_storageType == StorageType::dense)
                {
                    if (streamDescriptor.ignoreStream != nullptr)
                    {
                        RuntimeError("Dense input cannot have ignore label.");
                    }

                    // For dense sequence we use DenseSequenceDataIds.
                    DenseSequenceDataIdsPtr newSequenceDataDense = make_shared<DenseSequenceDataIds>();

                    array<int, 3> shape = *m_example->GetBlobShape(streamDescriptor.datasetName);
                    // Shape provided by image dataset has last dimension last, here we need last dimension first.
                    reverse(shape.begin(), shape.end());

                    // Move data from example to sequence. Although we reversed the shape we should not alter data since
                    // expected memory layout is the same (just shape notation is different).
                    m_example->SwapBlobData(streamDescriptor.datasetName, newSequenceDataDense->m_ownedData);
                    // Fill in the base class fields.
                    newSequenceDataDense->m_id = ismpl;
                    newSequenceDataDense->m_numberOfSamples = 1;
                    newSequenceDataDense->m_chunk = nullptr;
                    newSequenceDataDense->m_sampleLayout = make_shared<TensorShape>(shape[0], shape[1], shape[2]);

                    // Save new sequence in the set of sequences.
                    sequences.m_data[istr][ismpl] = newSequenceDataDense;

                    // Move to next stream.
                    istr++;
                }
                else
                {
                    vector<float>* ignoreBuffer = nullptr;
                    if (streamDescriptor.ignoreStream != nullptr)
                    {
                        if (istr + 1 >= m_inputStreams.size())
                        {
                            RuntimeError("Invalid number of input streams (sparse stream is not followed by ignore stream).");
                        }

                        // Dimensions of the output ignore tensor (height x width x 1, where values are 1 or 0 - zero
                        // means ignore classification result at that spatial position).
                        auto ignoreDims = m_inputStreams[istr + 1]->m_sampleLayout->GetDims();

                        DenseSequenceDataIdsPtr newSequenceDataDenseIgnore = make_shared<DenseSequenceDataIds>();

                        // We set all ignore output values to 1. Below we will set values to zero where output should be ignored.
                        newSequenceDataDenseIgnore->m_ownedData.assign(ignoreDims[0] * ignoreDims[1] * ignoreDims[2], 1);
                        newSequenceDataDenseIgnore->m_id = ismpl;
                        newSequenceDataDenseIgnore->m_numberOfSamples = 1;
                        newSequenceDataDenseIgnore->m_chunk = nullptr;
                        newSequenceDataDenseIgnore->m_sampleLayout = make_shared<TensorShape>(ignoreDims[0], ignoreDims[1], ignoreDims[2]);

                        // Save ignore sequence in the set of sequences.
                        sequences.m_data[istr + 1][ismpl] = newSequenceDataDenseIgnore;

                        // Save ignore buffer to be filled in below.
                        ignoreBuffer = &newSequenceDataDenseIgnore->m_ownedData;
                    }

                    // For sparse sequence we use SparseSequenceDataIds.
                    SparseSequenceDataIdsPtr newSequenceDataSparse = make_shared<SparseSequenceDataIds>();

                    auto dims = m_inputStreams[istr]->m_sampleLayout->GetDims();

                    // Move data from example to sequence.
                    vector<float> data;
                    m_example->SwapBlobData(streamDescriptor.datasetName, data);
                    if (data.size() != dims[0] * dims[1])
                    {
                        RuntimeError("Unexpected sparse data count");
                    }

                    newSequenceDataSparse->m_id = ismpl;
                    newSequenceDataSparse->m_numberOfSamples = 1;
                    newSequenceDataSparse->m_chunk = nullptr;

                    // We have data.size() non-zero values, set necessary counts.
                    newSequenceDataSparse->m_nnzCounts.resize(1);
                    newSequenceDataSparse->m_nnzCounts[0] = static_cast<IndexType>(data.size());
                    newSequenceDataSparse->m_totalNnzCount = static_cast<IndexType>(data.size());
                    // All the non-zero values are equal to 1.
                    newSequenceDataSparse->m_valuesMemory.resize(data.size(), 1.0f);
                    // Now we need to convert indices contained in data.
                    newSequenceDataSparse->m_indicesMemory.resize(data.size());
                    size_t spatialSize = dims[0] * dims[1];
                    // Out channels dimension is equal to number of outputs (we need distribution per class).
                    size_t outChannels = dims[2];
                    for (size_t inz = 0; inz < data.size(); inz++)
                    {
                        int c = static_cast<int>(data[inz]);

                        if (ignoreBuffer != nullptr && c == streamDescriptor.ignoreStream->ignoreLabel)
                        {
                            (*ignoreBuffer)[inz] = 0;
                            // In spite of ignoring this target we need to set some non-zero index for packer.
                            // This can be any index that corresponds to this output (inz which corresponds to
                            // 0 channel (0 class) is fine).
                            newSequenceDataSparse->m_indicesMemory[inz] = static_cast<IndexType>(inz);
                            continue;
                        }

                        // Here we need to be within the range.
                        if (c < 0 || c > outChannels - 1)
                        {
                            RuntimeError("Invalid channel value in sparse input stream.");
                        }

                        size_t index = c * spatialSize + inz;
                        newSequenceDataSparse->m_indicesMemory[inz] = static_cast<IndexType>(index);
                    }
                    newSequenceDataSparse->m_indices = newSequenceDataSparse->m_indicesMemory.data();

                    // Save new sequence in the set of sequences.
                    sequences.m_data[istr][ismpl] = newSequenceDataSparse;

                    // Move to the next stream.
                    istr++;
                    if (streamDescriptor.ignoreStream != nullptr)
                    {
                        // We had one additional stream (ignore mask).
                        istr++;
                    }
                }
            }
            // Move to the next example (sequence).
            m_dsLoader->GetExample(m_example.get());
        }

        m_currEpochSampleCount += static_cast<int>(sampleCount);

        return sequences;
    }

    bool UseAllExamplesFromDatasetForEpoch(const EpochConfiguration& config)
    {
        // If epoch size is equal to magic constant than we need to use all examples per epoch.
        return (config.m_totalEpochSizeInSamples == requestDataSize);
    }

private:

    // Input stream description describes the data coming out of dataset loader.
    vector<StreamDescriptionPtr> m_inputStreams;
    // Output stream description describes the data coming out of this object.
    vector<StreamDescriptionPtr> m_outputStreams;
    // Stream descriptions from the config.
    vector<StreamDescriptor> m_streamDescriptors;

    // Performs loading of the dataset.
    unique_ptr<IDsLoader<float> > m_dsLoader;
    // Object used for storing results from dataset loader.
    unique_ptr<ImageDatasetExample> m_example;

    // Size of the current minibatch.
    size_t m_minibatchSize;
    // Indicates if last minibatch is to be appended to next to last.
    bool m_appendLastMinibatch;
    // Distributed reading parameters.
    size_t m_workerRank;
    size_t m_numberOfWorkers;


    size_t m_epochSize;
    size_t m_currEpochSampleCount;

    // mIoU workaround: Forces readers to go through entire epoch (enables correct mIoU reporting).
    // TODO(VSO/OS/ANALOG_SL/#9673559): Remove workaround once proper mIoU reporting is implemented.
    bool m_epochOverride;
};

///////////////////////////////////////////////////////////////////////////////
//                      IMAGE DATASET READER IMPLEMENATION
///////////////////////////////////////////////////////////////////////////////

ImageDatasetReader::ImageDatasetReader(MemoryProviderPtr provider,
                                       const ConfigParameters& config)
{
    // Create data source and connect it to the packer.
    m_sequenceEnumerator = make_shared<DataSource>(config);

    m_packer = make_shared<FramePacker>(
        m_sequenceEnumerator,
        m_sequenceEnumerator->GetStreamDescriptions());
}

vector<StreamDescriptionPtr> ImageDatasetReader::GetStreamDescriptions()
{
    // Descriptions are saved in datasource, just forward the call.
    return m_sequenceEnumerator->GetStreamDescriptions();
}

} } }