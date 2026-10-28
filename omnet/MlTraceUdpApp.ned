import inet.applications.contract.IApp;

// Replays a packet trace over UDP. The trace is a CSV file with the columns
// timestamp_sec, size_bytes and flow_id (see align_and_generate_multi_ue.py).
simple MlTraceUdpApp like IApp
{
    parameters:
        string traceFile = default("ml_trace.csv");
        int localPort = default(-1);
        int destPort = default(1000);
        string destAddress;
        @display("i=block/app");
    gates:
        input socketIn;
        output socketOut;
}
