#include "inet/common/INETDefs.h"
#include "inet/applications/base/ApplicationBase.h"
#include "inet/transportlayer/contract/udp/UdpSocket.h"
#include "inet/common/packet/Packet.h"
#include "inet/common/packet/chunk/ByteCountChunk.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "inet/common/TimeTag_m.h"
#include "inet/applications/base/ApplicationPacket_m.h"

#include <fstream>
#include <string>
#include <sstream>

using namespace inet;

class MlTraceUdpApp : public ApplicationBase, public UdpSocket::ICallback {
  protected:
    UdpSocket socket;
    cMessage *timerEvent = nullptr;
    std::ifstream traceFile;

    simtime_t nextSendTime;
    int nextPacketSize;
    int nextFlowId;
    uint32_t numSent = 0;           // sequence number of the next packet

    virtual int numInitStages() const override { return NUM_INIT_STAGES; }
    virtual void initialize(int stage) override;
    virtual void handleMessageWhenUp(cMessage *msg) override;
    virtual void finish() override;

    bool readNextTraceLine();
    void sendPacket();

    // lifecycle (required by ApplicationBase)
    virtual void handleStartOperation(LifecycleOperation *operation) override {}
    virtual void handleStopOperation(LifecycleOperation *operation) override {}
    virtual void handleCrashOperation(LifecycleOperation *operation) override {}

    // UDP socket callbacks
    virtual void socketDataArrived(UdpSocket *socket, Packet *packet) override { delete packet; }
    virtual void socketErrorArrived(UdpSocket *socket, Indication *indication) override { delete indication; }
    virtual void socketClosed(UdpSocket *socket) override {}
};

Define_Module(MlTraceUdpApp);

void MlTraceUdpApp::initialize(int stage)
{
    ApplicationBase::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        timerEvent = new cMessage("sendTimer");

        // packet trace: CSV with the columns timestamp_sec, size_bytes, flow_id
        const char *filename = par("traceFile").stringValue();
        traceFile.open(filename);
        if (!traceFile.is_open()) {
            throw cRuntimeError("Cannot open trace file: %s", filename);
        }

        // skip the header row
        std::string header;
        std::getline(traceFile, header);

    } else if (stage == INITSTAGE_APPLICATION_LAYER) {
        // UDP socket
        socket.setOutputGate(gate("socketOut"));
        socket.bind(par("localPort"));
        
        L3Address destAddr;
        L3AddressResolver().tryResolve(par("destAddress").stringValue(), destAddr);
        socket.connect(destAddr, par("destPort"));

        // schedule the first packet of the trace
        if (readNextTraceLine()) {
            scheduleAt(nextSendTime, timerEvent);
        }
    }
}

bool MlTraceUdpApp::readNextTraceLine()
{
    std::string line;
    if (std::getline(traceFile, line)) {
        std::stringstream ss(line);
        std::string item;

        std::getline(ss, item, ',');
        nextSendTime = std::stod(item);

        std::getline(ss, item, ',');
        nextPacketSize = std::stoi(item);

        std::getline(ss, item, ',');
        nextFlowId = std::stoi(item);

        return true;
    }
    return false;   // end of the trace
}

void MlTraceUdpApp::handleMessageWhenUp(cMessage *msg)
{
    if (msg == timerEvent) {
        sendPacket();

        // schedule the next packet
        if (readNextTraceLine()) {
            scheduleAt(nextSendTime, timerEvent);
        }
    } else {
        socket.processMessage(msg);
    }
}

void MlTraceUdpApp::sendPacket()
{
    int remainingSize = nextPacketSize;
    int maxPayload = 1400;   // larger packets are sent as several datagrams, without IP fragmentation
    int fragIdx = 0;

    while (remainingSize > 0) {
        int currentSize = std::min(remainingSize, maxPayload);
        remainingSize -= currentSize;

        std::string pktName = "ML_Flow_" + std::to_string(nextFlowId) + "_part" + std::to_string(fragIdx++);
        Packet *packet = new Packet(pktName.c_str());
        
        // INET application packet with a creation-time tag, as in UdpBasicApp
        const auto& payload = makeShared<ApplicationPacket>();
        payload->setChunkLength(B(currentSize));
        payload->setSequenceNumber(numSent++);
        
        payload->addTagIfAbsent<CreationTimeTag>()->setCreationTime(simTime());

        packet->insertAtBack(payload);

        EV_INFO << "Injecting Trace Packet: " << pktName << " | Size: " << currentSize << "B\n";
        
        socket.send(packet);
    }
}

void MlTraceUdpApp::finish()
{
    ApplicationBase::finish();
    cancelAndDelete(timerEvent);
    if (traceFile.is_open()) traceFile.close();
}