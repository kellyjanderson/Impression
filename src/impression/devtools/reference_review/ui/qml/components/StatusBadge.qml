import QtQuick
import QtQuick.Controls

Control {
    id: root
    property string label: ""
    property string tone: "neutral"

    implicitHeight: 28
    implicitWidth: Math.max(96, contentItem.implicitWidth + leftPadding + rightPadding)
    leftPadding: 10
    rightPadding: 10
    topPadding: 4
    bottomPadding: 4

    contentItem: Text {
        text: root.label
        color: root.tone === "warning" ? "#5c360f" : root.tone === "ready" ? "#1f4f52" : "#42453f"
        font.pixelSize: 12
        font.bold: true
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }

    background: Rectangle {
        radius: 4
        color: root.tone === "warning" ? "#ffe2b8" : root.tone === "ready" ? "#d6eeee" : "#ecece4"
        border.color: root.tone === "warning" ? "#d9a457" : root.tone === "ready" ? "#90bdc1" : "#c9c8be"
    }
}
