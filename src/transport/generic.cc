#include "comm.h"
#include "transport.h"
#include "bootstrap.h"

// 主要就是调用ncclTransportP2pConnect和ncclTransportP2pSetup，
// 而ncclTransportP2pConnect也不是真的把P2P channel的两个rank之间连接起来，
// 而是根据当前rank在每个channel中的prev和next，设置对应rank的mask，
// 后续真正建立transport时根据这里的mask决定哪些rank之间需要创建，以及需要创建多少。
ncclResult_t ncclTransportRingConnect(struct ncclComm* comm) {
  struct ringConnInfo {
    bool useNetPXN;
    bool useGdr;
  };
  struct ringConnInfo* ringInfo = NULL;
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    comm->useGdr = true;
    comm->useNetPXN = false;
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, fail);
    }
    // 关键的还是ncclTransportP2pSetup这个函数，这个函数是创建transport的核心。
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0), ret, fail);
    if (ncclParamLocalRegister() || ncclParamGraphRegister()) {
      NCCLCHECK(ncclCalloc(&ringInfo, comm->nRanks));
      ringInfo[comm->rank].useGdr = comm->useGdr;
      ringInfo[comm->rank].useNetPXN = comm->useNetPXN;
      NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ringInfo, sizeof(struct ringConnInfo)), ret, fail);
      for (int i = 0; i < comm->nRanks; ++i) {
        if (!ringInfo[i].useGdr) comm->useGdr = false;
        if (ringInfo[i].useNetPXN) comm->useNetPXN = true;
        if (comm->useGdr == false && comm->useNetPXN == true) break;
      }
    }
    INFO(NCCL_INIT, "Connected all rings, use ring PXN %d GDR %d", comm->useNetPXN, comm->useGdr);
  }
exit:
  free(ringInfo);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportTreeConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    // Connect Trees
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0), ret, fail);
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    INFO(NCCL_INIT, "Connected all trees");
  }
exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportPatConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    for (int mask=1; mask<comm->nRanks; mask<<=1) {
      int prevPeer = (comm->rank + mask) % comm->nRanks;
      int nextPeer = (comm->rank + comm->nRanks - mask) % comm->nRanks;
      for (int c = 0; c < comm->nChannels; c++) {
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &prevPeer, 1, &nextPeer, 0), ret, fail); // ReduceScatter
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
      for (int c = 0; c < comm->nChannels; c++) {
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &nextPeer, 1, &prevPeer, 0), ret, fail); // AllGather
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    }
    INFO(NCCL_INIT, "Connected binomial trees");
  }
exit:
  return ret;
fail:
  goto exit;
}
