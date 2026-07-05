import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "components" as Components

ApplicationWindow {
    id: root
    width: 1180
    height: 760
    minimumWidth: 880
    minimumHeight: 560
    visible: true
    title: "Reference Review Workbench"
    color: "#f7f7f2"

    SplitView {
        anchors.fill: parent
        orientation: Qt.Horizontal

        Frame {
            SplitView.preferredWidth: 280
            SplitView.minimumWidth: 220
            SplitView.maximumWidth: 360
            padding: 12

            ColumnLayout {
                anchors.fill: parent
                spacing: 10

                Text {
                    Layout.fillWidth: true
                    text: "Queue"
                    font.pixelSize: 18
                    font.bold: true
                    color: "#242622"
                    elide: Text.ElideRight
                }

                Components.StatusBadge {
                    Layout.fillWidth: true
                    label: "No fixture loaded"
                    tone: "neutral"
                }

                ListView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    model: 0
                    delegate: ItemDelegate {
                        width: ListView.view.width
                        height: 44
                        text: model.display
                    }
                }
            }
        }

        Frame {
            SplitView.fillWidth: true
            padding: 16

            ColumnLayout {
                anchors.fill: parent
                spacing: 12

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    Text {
                        Layout.fillWidth: true
                        text: "Selected Fixture"
                        font.pixelSize: 18
                        font.bold: true
                        color: "#242622"
                        elide: Text.ElideRight
                    }

                    Components.StatusBadge {
                        label: startupDiagnostics.length > 0 ? "Diagnostics" : "Ready"
                        tone: startupDiagnostics.length > 0 ? "warning" : "ready"
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    radius: 4
                    border.color: "#c9c8be"
                    color: "#ffffff"

                    Label {
                        anchors.centerIn: parent
                        width: Math.min(parent.width - 48, 520)
                        text: "Select a fixture to begin review."
                        horizontalAlignment: Text.AlignHCenter
                        wrapMode: Text.WordWrap
                        color: "#565a51"
                    }
                }
            }
        }
    }
}
