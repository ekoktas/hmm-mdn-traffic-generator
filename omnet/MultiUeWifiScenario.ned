import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.ethernet.Eth100M;
import inet.node.inet.StandardHost;
import inet.node.inet.WirelessHost;
import inet.node.wireless.AccessPoint;
import inet.physicallayer.wireless.ieee80211.packetlevel.Ieee80211ScalarRadioMedium;

// Five UEs share one IEEE 802.11 channel with an access point; the access point is
// connected to the server by 100 Mbit/s Ethernet. Positions and Wi-Fi settings: omnetpp.ini.
network MultiUeWifiScenario
{
    parameters:
        int numUes = default(5);
    submodules:
        configurator: Ipv4NetworkConfigurator { @display("p=100,100"); }
        radioMedium: Ieee80211ScalarRadioMedium { @display("p=100,200"); }
        ap: AccessPoint { @display("p=400,300"); }
        ue[numUes]: WirelessHost { @display("p=200,100"); }
        server: StandardHost { @display("p=600,300"); }
    connections:
        server.ethg++ <--> Eth100M <--> ap.ethg++;
}
