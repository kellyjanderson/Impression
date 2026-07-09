import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Frame {
    id: root
    property int artifactCount: 0
    property var artifacts: []

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
            model: root.artifacts.length > 0 ? root.artifacts : root.artifactCount

            delegate: Rectangle {
                width: 148
                height: 92
                radius: 4
                border.color: "#c9c8be"
                color: "#ffffff"

                Column {
                    anchors.fill: parent
                    anchors.margins: 6
                    spacing: 4

                    Image {
                        width: parent.width
                        height: 56
                        source: typeof modelData === "object" ? modelData.artifact_preview_url : ""
                        fillMode: Image.PreserveAspectFit
                        asynchronous: true
                        cache: false
                        visible: source !== "" && status === Image.Ready
                    }

                    Rectangle {
                        width: parent.width
                        height: 56
                        radius: 3
                        color: "#f7f7f2"
                        border.color: "#d8d7cf"
                        visible: typeof modelData !== "object" || modelData.artifact_preview_url === ""

                        Text {
                            anchors.centerIn: parent
                            width: parent.width - 10
                            text: "No preview"
                            horizontalAlignment: Text.AlignHCenter
                            elide: Text.ElideRight
                            color: "#565a51"
                            font.pixelSize: 11
                        }
                    }

                    Text {
                        width: parent.width
                        text: typeof modelData === "object" ? modelData.artifact_display_path : "Artifact"
                        horizontalAlignment: Text.AlignHCenter
                        elide: Text.ElideMiddle
                        color: "#565a51"
                        font.pixelSize: 11
                    }
                }
            }
        }
    }
}
