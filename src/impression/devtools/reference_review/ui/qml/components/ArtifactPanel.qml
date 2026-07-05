import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Frame {
    id: root
    property int artifactCount: 0

    padding: 12

    ColumnLayout {
        anchors.fill: parent
        spacing: 8

        Text {
            Layout.fillWidth: true
            text: "Artifacts"
            font.pixelSize: 16
            font.bold: true
            color: "#242622"
            elide: Text.ElideRight
        }

        GridView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            cellWidth: 160
            cellHeight: 104
            clip: true
            model: root.artifactCount

            delegate: Rectangle {
                width: 148
                height: 92
                radius: 4
                border.color: "#c9c8be"
                color: "#ffffff"

                Text {
                    anchors.centerIn: parent
                    width: parent.width - 16
                    text: "Artifact"
                    horizontalAlignment: Text.AlignHCenter
                    elide: Text.ElideRight
                    color: "#565a51"
                }
            }
        }
    }
}
